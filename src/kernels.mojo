"""Compute kernels for embedding training, similarity search, and sparse LSI."""

from std.math import exp, sqrt
from std.sys.info import simd_width_of

comptime W = simd_width_of[DType.float32]()
comptime FPtr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime DPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime PARALLEL_WORK = 200_000
comptime WORKERS = 16
comptime PARALLEL_TASKS = WORKERS * 4


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def dp(addr: Int) -> DPtr:
    return DPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


@always_inline
def dot(a: FPtr, b: FPtr, n: Int) -> Float32:
    var vacc = SIMD[DType.float32, W](0.0)
    var i = 0
    while i + W <= n:
        vacc += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var total = vacc.reduce_add()
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


@always_inline
def sigmoid(value: Float32) -> Float32:
    if value > 15.0:
        return 0.9999997
    if value < -15.0:
        return 0.0000003
    return 1.0 / (1.0 + exp(-value))


@export("mg_normalize_rows")
def mg_normalize_rows(
    src_addr: Int, dst_addr: Int, rows: Int, cols: Int
) abi("C"):
    var src = fp(src_addr)
    var dst = fp(dst_addr)

    def normalize_row(row: Int) {imm}:
        var base = row * cols
        var norm = sqrt(max(dot(src + base, src + base, cols), Float32(0.0)))
        if norm == 0.0:
            for col in range(cols):
                dst[base + col] = 0.0
        else:
            var inverse = 1.0 / norm
            var col = 0
            while col + W <= cols:
                dst.store(
                    base + col,
                    src.load[width=W](base + col)
                    * SIMD[DType.float32, W](inverse),
                )
                col += W
            while col < cols:
                dst[base + col] = src[base + col] * inverse
                col += 1

    def normalize_chunk(task: Int) {imm}:
        var begin = rows * task // PARALLEL_TASKS
        var end = rows * (task + 1) // PARALLEL_TASKS
        for row in range(begin, end):
            normalize_row(row)

    if rows * cols >= PARALLEL_WORK:
        for task in range(PARALLEL_TASKS):
            normalize_chunk(task)
    else:
        for row in range(rows):
            normalize_row(row)


@export("mg_dot_rows")
def mg_dot_rows(
    matrix_addr: Int,
    query_addr: Int,
    scores_addr: Int,
    rows: Int,
    cols: Int,
) abi("C"):
    var matrix = fp(matrix_addr)
    var query = fp(query_addr)
    var scores = fp(scores_addr)

    def score_row(row: Int) {imm}:
        scores[row] = dot(matrix + row * cols, query, cols)

    def score_chunk(task: Int) {imm}:
        var begin = rows * task // PARALLEL_TASKS
        var end = rows * (task + 1) // PARALLEL_TASKS
        for row in range(begin, end):
            score_row(row)

    if rows * cols >= PARALLEL_WORK:
        for task in range(PARALLEL_TASKS):
            score_chunk(task)
    else:
        for row in range(rows):
            score_row(row)


@export("mg_cosine_rows")
def mg_cosine_rows(
    matrix_addr: Int,
    norms_addr: Int,
    query_addr: Int,
    scores_addr: Int,
    rows: Int,
    cols: Int,
) abi("C"):
    var matrix = fp(matrix_addr)
    var norms = fp(norms_addr)
    var query = fp(query_addr)
    var scores = fp(scores_addr)

    def score_row(row: Int) {imm}:
        var norm = norms[row]
        if norm == 0.0:
            scores[row] = 0.0
        else:
            scores[row] = dot(matrix + row * cols, query, cols) / norm

    def score_chunk(task: Int) {imm}:
        var begin = rows * task // PARALLEL_TASKS
        var end = rows * (task + 1) // PARALLEL_TASKS
        for row in range(begin, end):
            score_row(row)

    if rows * cols >= PARALLEL_WORK:
        for task in range(PARALLEL_TASKS):
            score_chunk(task)
    else:
        for row in range(rows):
            score_row(row)


@export("mg_sg_pairs")
def mg_sg_pairs(
    tokens_addr: Int,
    indptr_addr: Int,
    reduced_addr: Int,
    centers_addr: Int,
    targets_addr: Int,
    sentence_count: Int,
    window: Int,
) abi("C") -> Int:
    var tokens = ip(tokens_addr)
    var indptr = ip(indptr_addr)
    var reduced = ip(reduced_addr)
    var centers = ip(centers_addr)
    var targets = ip(targets_addr)
    var pair = 0
    for sentence in range(sentence_count):
        var begin = Int(indptr[sentence])
        var end = Int(indptr[sentence + 1])
        for position in range(begin, end):
            var shrink = Int(reduced[position])
            var left = max(begin, position - window + shrink)
            var right = min(end, position + window + 1 - shrink)
            for context in range(left, right):
                if context != position:
                    centers[pair] = tokens[position]
                    targets[pair] = tokens[context]
                    pair += 1
    return pair


@export("mg_sgns_train")
def mg_sgns_train(
    input_addr: Int,
    output_addr: Int,
    centers_addr: Int,
    targets_addr: Int,
    negatives_addr: Int,
    pair_count: Int,
    dims: Int,
    negative_count: Int,
    alpha: Float32,
) abi("C"):
    var inputs = fp(input_addr)
    var outputs = fp(output_addr)
    var centers = ip(centers_addr)
    var targets = ip(targets_addr)
    var negatives = ip(negatives_addr)
    for pair in range(pair_count):
        var center = Int(centers[pair])
        var input_base = center * dims
        for sample in range(negative_count + 1):
            var target = Int(targets[pair])
            var label = Float32(1.0)
            if sample > 0:
                target = Int(negatives[pair * negative_count + sample - 1])
                label = 0.0
            var output_base = target * dims
            var gradient = (label - sigmoid(
                dot(inputs + input_base, outputs + output_base, dims)
            )) * alpha
            var gradient_vector = SIMD[DType.float32, W](gradient)
            var dim = 0
            while dim + W <= dims:
                var input_values = inputs.load[width=W](input_base + dim)
                var output_values = outputs.load[width=W](output_base + dim)
                inputs.store(
                    input_base + dim,
                    input_values + gradient_vector * output_values,
                )
                outputs.store(
                    output_base + dim,
                    output_values + gradient_vector * input_values,
                )
                dim += W
            while dim < dims:
                var input_value = inputs[input_base + dim]
                var output_value = outputs[output_base + dim]
                inputs[input_base + dim] = input_value + gradient * output_value
                outputs[output_base + dim] = output_value + gradient * input_value
                dim += 1


@export("mg_cbow_train")
def mg_cbow_train(
    input_addr: Int,
    output_addr: Int,
    context_addr: Int,
    indptr_addr: Int,
    targets_addr: Int,
    negatives_addr: Int,
    hidden_addr: Int,
    gradient_addr: Int,
    item_count: Int,
    dims: Int,
    negative_count: Int,
    alpha: Float32,
    use_mean: Int,
) abi("C"):
    var inputs = fp(input_addr)
    var outputs = fp(output_addr)
    var contexts = ip(context_addr)
    var indptr = ip(indptr_addr)
    var targets = ip(targets_addr)
    var negatives = ip(negatives_addr)
    var hidden = fp(hidden_addr)
    var error = fp(gradient_addr)
    for item in range(item_count):
        var begin = Int(indptr[item])
        var end = Int(indptr[item + 1])
        var count = end - begin
        for dim in range(dims):
            hidden[dim] = 0.0
            error[dim] = 0.0
        for pos in range(begin, end):
            var input_base = Int(contexts[pos]) * dims
            for dim in range(dims):
                hidden[dim] += inputs[input_base + dim]
        if use_mean != 0 and count > 0:
            var inverse = 1.0 / Float32(count)
            for dim in range(dims):
                hidden[dim] *= inverse
        for sample in range(negative_count + 1):
            var target = Int(targets[item])
            var label = Float32(1.0)
            if sample > 0:
                target = Int(negatives[item * negative_count + sample - 1])
                label = 0.0
            var output_base = target * dims
            var update = (label - sigmoid(
                dot(hidden, outputs + output_base, dims)
            )) * alpha
            for dim in range(dims):
                var output_value = outputs[output_base + dim]
                error[dim] += update * output_value
                outputs[output_base + dim] = output_value + update * hidden[dim]
        if use_mean != 0 and count > 0:
            var inverse = 1.0 / Float32(count)
            for dim in range(dims):
                error[dim] *= inverse
        for pos in range(begin, end):
            var input_base = Int(contexts[pos]) * dims
            for dim in range(dims):
                inputs[input_base + dim] += error[dim]


@export("mg_dm_train")
def mg_dm_train(
    word_addr: Int,
    doc_addr: Int,
    output_addr: Int,
    context_addr: Int,
    indptr_addr: Int,
    docs_addr: Int,
    targets_addr: Int,
    negatives_addr: Int,
    hidden_addr: Int,
    gradient_addr: Int,
    item_count: Int,
    dims: Int,
    negative_count: Int,
    alpha: Float32,
    use_mean: Int,
    update_words: Int,
) abi("C"):
    var words = fp(word_addr)
    var docs = fp(doc_addr)
    var outputs = fp(output_addr)
    var contexts = ip(context_addr)
    var indptr = ip(indptr_addr)
    var doc_ids = ip(docs_addr)
    var targets = ip(targets_addr)
    var negatives = ip(negatives_addr)
    var hidden = fp(hidden_addr)
    var error = fp(gradient_addr)
    for item in range(item_count):
        var begin = Int(indptr[item])
        var end = Int(indptr[item + 1])
        var count = end - begin + 1
        var doc_base = Int(doc_ids[item]) * dims
        for dim in range(dims):
            hidden[dim] = docs[doc_base + dim]
            error[dim] = 0.0
        for pos in range(begin, end):
            var word_base = Int(contexts[pos]) * dims
            for dim in range(dims):
                hidden[dim] += words[word_base + dim]
        if use_mean != 0:
            var inverse = 1.0 / Float32(count)
            for dim in range(dims):
                hidden[dim] *= inverse
        for sample in range(negative_count + 1):
            var target = Int(targets[item])
            var label = Float32(1.0)
            if sample > 0:
                target = Int(negatives[item * negative_count + sample - 1])
                label = 0.0
            var output_base = target * dims
            var update = (label - sigmoid(
                dot(hidden, outputs + output_base, dims)
            )) * alpha
            for dim in range(dims):
                var output_value = outputs[output_base + dim]
                error[dim] += update * output_value
                outputs[output_base + dim] = output_value + update * hidden[dim]
        if use_mean != 0:
            var inverse = 1.0 / Float32(count)
            for dim in range(dims):
                error[dim] *= inverse
        for dim in range(dims):
            docs[doc_base + dim] += error[dim]
        if update_words != 0:
            for pos in range(begin, end):
                var word_base = Int(contexts[pos]) * dims
                for dim in range(dims):
                    words[word_base + dim] += error[dim]


@export("mg_csr_matmul")
def mg_csr_matmul(
    indptr_addr: Int,
    indices_addr: Int,
    values_addr: Int,
    right_addr: Int,
    result_addr: Int,
    rows: Int,
    cols: Int,
    width: Int,
) abi("C"):
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var values = dp(values_addr)
    var right = dp(right_addr)
    var result = dp(result_addr)

    def multiply_row(row: Int) {imm}:
        var result_base = row * width
        for col in range(width):
            result[result_base + col] = 0.0
        for pos in range(Int(indptr[row]), Int(indptr[row + 1])):
            var source = Int(indices[pos])
            var value = values[pos]
            for col in range(width):
                result[result_base + col] += value * right[source * width + col]

    if rows * width >= 4096:
        for row in range(rows):
            multiply_row(row)
    else:
        for row in range(rows):
            multiply_row(row)


@export("mg_csr_t_matmul")
def mg_csr_t_matmul(
    indptr_addr: Int,
    indices_addr: Int,
    values_addr: Int,
    right_addr: Int,
    result_addr: Int,
    rows: Int,
    cols: Int,
    width: Int,
) abi("C"):
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var values = dp(values_addr)
    var right = dp(right_addr)
    var result = dp(result_addr)
    for i in range(cols * width):
        result[i] = 0.0
    for row in range(rows):
        var right_base = row * width
        for pos in range(Int(indptr[row]), Int(indptr[row + 1])):
            var target_base = Int(indices[pos]) * width
            var value = values[pos]
            for col in range(width):
                result[target_base + col] += value * right[right_base + col]
