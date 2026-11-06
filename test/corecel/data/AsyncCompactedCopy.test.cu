//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AsyncCompactedCopy.test.cu
//---------------------------------------------------------------------------//
#include "AsyncCompactedCopy.test.hh"

#include "corecel/data/AsyncCompactedCopy.device.t.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Stream.hh"

namespace celeritas
{
template class AsyncCompactedCopy<test::CompactTestItem>;

namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Compact and copy batches of equally sized items as a stepping loop would.
 *
 * Each batch represents the per-track-slot items of a step. As in the optical
 * transport loop, the host waits for each step before starting the next, and
 * the items of a step are processed two steps later.
 */
VecVecInt compacted_copy_test(VecVecInt const& batches)
{
    CELER_EXPECT(!batches.empty());

    StreamId const stream_id{0};
    auto& stream = device().stream(stream_id);
    auto const size = static_cast<size_type>(batches.front().size());

    AsyncCompactedCopy<CompactTestItem> copier(size, stream_id);
    DeviceVector<CompactTestItem> items(size);

    VecVecInt result;
    auto take_copied = [&] {
        auto& out = result.emplace_back();
        for (auto const& item : copier.take_copied())
        {
            out.push_back(item.value);
        }
    };

    for (auto const& batch : batches)
    {
        CELER_ASSERT(batch.size() == size);

        // Wait for the previous step, as the transport loop does
        stream.sync();
        if (copier.has_compacted())
        {
            if (copier.has_copied())
            {
                take_copied();
            }
            copier.copy_async();
        }

        // Write the items for this step and compact them
        std::vector<CompactTestItem> host_items;
        for (int v : batch)
        {
            host_items.push_back(CompactTestItem{v});
        }
        items.copy_to_device(make_span(host_items));
        copier.compact_async(make_observer(items), size);
    }

    // Process the remaining items in order
    if (copier.has_copied())
    {
        take_copied();
    }
    if (copier.has_compacted())
    {
        copier.copy_async();
        take_copied();
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Assign to a default-constructed copier, then replace its storage.
 *
 * Each batch has a different size and is taken before the next assignment.
 */
VecVecInt assign_test(VecVecInt const& batches)
{
    StreamId const stream_id{0};

    AsyncCompactedCopy<CompactTestItem> copier;

    VecVecInt result;
    for (auto const& batch : batches)
    {
        auto const size = static_cast<size_type>(batch.size());
        copier = AsyncCompactedCopy<CompactTestItem>(size, stream_id);
        CELER_ASSERT(copier.capacity() == size);

        std::vector<CompactTestItem> host_items;
        for (int v : batch)
        {
            host_items.push_back(CompactTestItem{v});
        }
        DeviceVector<CompactTestItem> items(size);
        items.copy_to_device(make_span(host_items));

        copier.compact_async(make_observer(items), size);
        copier.copy_async();

        auto& out = result.emplace_back();
        for (auto const& item : copier.take_copied())
        {
            out.push_back(item.value);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Assign to a copier while a compaction and a copy are pending.
 *
 * This throws if assertions are enabled. Otherwise, the old storage must
 * outlive the pending work, and the new storage must be usable.
 */
void assign_pending_test()
{
    StreamId const stream_id{0};
    size_type const size = 1024;

    DeviceVector<CompactTestItem> items(size);
    std::vector<CompactTestItem> host_items(size, CompactTestItem{1});
    items.copy_to_device(make_span(host_items));

    AsyncCompactedCopy<CompactTestItem> copier(size, stream_id);
    copier.compact_async(make_observer(items), size);
    copier.copy_async();
    copier.compact_async(make_observer(items), size);

    copier = AsyncCompactedCopy<CompactTestItem>(size, stream_id);
    CELER_VALIDATE(!copier.has_compacted() && !copier.has_copied(),
                   << "assignment should replace the pipeline state");

    copier.compact_async(make_observer(items), size);
    copier.copy_async();
    CELER_VALIDATE(copier.take_copied().size() == size,
                   << "new storage should hold all items");
}

//---------------------------------------------------------------------------//
/*!
 * Destroy a copier while a compaction and a copy are pending.
 *
 * The items of the first step are being copied and those of the second step
 * are being compacted, as when an exception unwinds the transport loop.
 */
void destroy_pending_test()
{
    StreamId const stream_id{0};
    size_type const size = 1024;

    DeviceVector<CompactTestItem> items(size);
    std::vector<CompactTestItem> host_items(size, CompactTestItem{1});
    items.copy_to_device(make_span(host_items));

    AsyncCompactedCopy<CompactTestItem> copier(size, stream_id);
    copier.compact_async(make_observer(items), size);
    copier.copy_async();
    copier.compact_async(make_observer(items), size);
    CELER_ASSERT(copier.has_compacted() && copier.has_copied());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
