#ifndef ASCEND_SPMM_PACKED_COLUMN_REINDEX_H_
#define ASCEND_SPMM_PACKED_COLUMN_REINDEX_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ascend_spmm {

// Convert original node ids to packed row ids.  If packed = P * original,
// replacing every column c with inverse_permutation[c] represents P*A*P^T.
template <typename index_t>
std::vector<index_t> renumber_columns_for_packed_layout(
    const std::vector<index_t>& columns,
    const std::vector<int64_t>& inverse_permutation)
{
    std::vector<index_t> renumbered(columns.size());
    for (std::size_t i = 0; i < columns.size(); ++i) {
        const int64_t column = static_cast<int64_t>(columns[i]);
        if (column < 0 ||
            static_cast<std::size_t>(column) >= inverse_permutation.size()) {
            throw std::out_of_range(
                "column id cannot be mapped into the packed row layout");
        }
        const int64_t packed_column =
            inverse_permutation[static_cast<std::size_t>(column)];
        if (packed_column < 0) {
            throw std::invalid_argument(
                "inverse permutation does not map every original column");
        }
        if (static_cast<uint64_t>(packed_column) >
            static_cast<uint64_t>(std::numeric_limits<index_t>::max())) {
            throw std::overflow_error(
                "packed column id does not fit the sparse index type");
        }
        renumbered[i] = static_cast<index_t>(packed_column);
    }
    return renumbered;
}

template <typename index_t, typename value_t>
void transpose_into_packed_layout(
    const std::vector<index_t>& row_pointer,
    const std::vector<index_t>& columns,
    const std::vector<value_t>& values,
    const std::vector<int64_t>& inverse_permutation,
    std::size_t packed_rows,
    std::vector<index_t>& transposed_row_pointer,
    std::vector<index_t>& transposed_columns,
    std::vector<value_t>& transposed_values)
{
    const std::size_t rows = inverse_permutation.size();
    if (row_pointer.size() != rows + 1 || columns.size() != values.size()) {
        throw std::invalid_argument("invalid CSR arrays for packed transpose");
    }
    if (static_cast<std::size_t>(row_pointer.back()) != columns.size()) {
        throw std::invalid_argument("CSR row pointer does not end at nnz");
    }

    transposed_row_pointer.assign(packed_rows + 1, index_t(0));
    for (const index_t column_value : columns) {
        const int64_t original_column = static_cast<int64_t>(column_value);
        if (original_column < 0 ||
            static_cast<std::size_t>(original_column) >= rows) {
            throw std::out_of_range("transpose column is outside the graph");
        }
        const int64_t packed_row =
            inverse_permutation[static_cast<std::size_t>(original_column)];
        if (packed_row < 0 || static_cast<std::size_t>(packed_row) >= packed_rows) {
            throw std::invalid_argument("invalid packed row in inverse permutation");
        }
        ++transposed_row_pointer[static_cast<std::size_t>(packed_row) + 1];
    }
    for (std::size_t row = 0; row < packed_rows; ++row) {
        transposed_row_pointer[row + 1] += transposed_row_pointer[row];
    }

    transposed_columns.resize(columns.size());
    transposed_values.resize(values.size());
    std::vector<index_t> next = transposed_row_pointer;
    for (std::size_t original_row = 0; original_row < rows; ++original_row) {
        const int64_t packed_column = inverse_permutation[original_row];
        if (packed_column < 0 || static_cast<std::size_t>(packed_column) >= packed_rows) {
            throw std::invalid_argument("invalid packed column in inverse permutation");
        }
        const std::size_t begin = static_cast<std::size_t>(row_pointer[original_row]);
        const std::size_t end = static_cast<std::size_t>(row_pointer[original_row + 1]);
        if (begin > end || end > columns.size()) {
            throw std::invalid_argument("non-monotonic CSR row pointer");
        }
        for (std::size_t position = begin; position < end; ++position) {
            const std::size_t packed_row = static_cast<std::size_t>(
                inverse_permutation[static_cast<std::size_t>(columns[position])]);
            const std::size_t destination = static_cast<std::size_t>(next[packed_row]++);
            transposed_columns[destination] = static_cast<index_t>(packed_column);
            transposed_values[destination] = values[position];
        }
    }
}

}  // namespace ascend_spmm

#endif  // ASCEND_SPMM_PACKED_COLUMN_REINDEX_H_
