#ifndef ASCEND_SPMM_BOUNDARY_BUCKET_SORT_H_
#define ASCEND_SPMM_BOUNDARY_BUCKET_SORT_H_

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ascend_spmm {

// Stable counting buckets for integer occupancies in [1, max_occupancy].
// Complexity: O(columns.size() + max_occupancy) time and auxiliary space.
template <typename index_t>
void stable_bucket_sort_by_occupancy_desc(
    std::vector<index_t>& columns,
    const std::vector<int64_t>& occupancy,
    int64_t max_occupancy)
{
    if (max_occupancy <= 0) {
        if (columns.empty()) return;
        throw std::invalid_argument("max occupancy must be positive");
    }

    std::vector<std::size_t> bucket_offsets(
        static_cast<std::size_t>(max_occupancy) + 1, 0);
    for (const index_t column : columns) {
        const int64_t column_id = static_cast<int64_t>(column);
        if (column_id < 0 ||
            static_cast<std::size_t>(column_id) >= occupancy.size()) {
            throw std::out_of_range("column id is outside the occupancy array");
        }
        const int64_t count = occupancy[static_cast<std::size_t>(column_id)];
        if (count <= 0 || count > max_occupancy) {
            throw std::invalid_argument(
                "block occupancy must be in [1, window_size]; coalesce duplicate CSR entries");
        }
        ++bucket_offsets[static_cast<std::size_t>(count)];
    }

    std::size_t offset = 0;
    for (int64_t count = max_occupancy; count >= 1; --count) {
        const std::size_t bucket = static_cast<std::size_t>(count);
        const std::size_t bucket_size = bucket_offsets[bucket];
        bucket_offsets[bucket] = offset;
        offset += bucket_size;
    }

    std::vector<index_t> sorted(columns.size());
    for (const index_t column : columns) {
        const std::size_t column_id = static_cast<std::size_t>(column);
        const std::size_t bucket =
            static_cast<std::size_t>(occupancy[column_id]);
        sorted[bucket_offsets[bucket]++] = column;
    }
    columns.swap(sorted);
}

}  // namespace ascend_spmm

#endif  // ASCEND_SPMM_BOUNDARY_BUCKET_SORT_H_
