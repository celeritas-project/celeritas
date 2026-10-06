//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/LocalProcessorSlots.hh
//! \sa test/celeritas/ext/GeantSd.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Per-stream cache of thread-local processors owned by local transporters.
 *
 * Classes that interface Celeritas with thread-local Geant4 objects (e.g.,
 * sensitive detectors) are shared across threads but must dispatch to a
 * processor that is created, used, and destroyed on a single worker thread.
 * This class holds one weakly referenced processor per stream.
 *
 * The references between a slot and its processor are cyclic and
 * deliberately non-owning in both directions:
 * \verbatim
   Shared --(shared)--> Slot --(weak + raw cache)--> Processor
                         ^                              |
                         +--(weak, via custom deleter)--+

   LocalTransporter --(shared, with custom deleter)--> Processor
   \endverbatim
 *
 * The thread-local transporter shares ownership of the processor, and the
 * shared object owns the slot. Since the cross references are weak, either
 * side may be destroyed first:
 * - when the last local reference to the processor is released (on the
 *   worker thread that created it), the deleter resets the slot's cached
 *   pointers so a later \c make call recreates the processor instead of
 *   returning a dangling pointer;
 * - when the slots are destroyed first, locking the weak pointer fails and
 *   only the processor is deleted.
 *
 * The weak pointer is used only when creating or recreating a local
 * processor. Step processing uses the cached raw pointer to avoid locking a
 * weak pointer on every step iteration.
 */
template<class T>
class LocalProcessorSlots
{
  public:
    //!@{
    //! \name Type aliases
    using SPProcessor = std::shared_ptr<T>;
    //!@}

  public:
    //! Construct without any streams
    LocalProcessorSlots() = default;

    // Construct with the number of streams
    explicit inline LocalProcessorSlots(StreamId::size_type num_streams);

    // Get or create the processor for a stream
    template<class F>
    inline SPProcessor make(StreamId sid, F&& construct);

    // Access the processor for a stream
    inline T& get(StreamId sid) const;

    //! Number of streams
    StreamId::size_type size() const { return slots_.size(); }

  private:
    struct Slot
    {
        std::weak_ptr<T> weak_processor;
        T* processor{nullptr};
    };

    struct SlotDeleter
    {
        std::weak_ptr<Slot> weak_slot;

        void operator()(T* processor) const
        {
            if (auto slot = weak_slot.lock())
            {
                if (slot->processor == processor)
                {
                    slot->processor = nullptr;
                    slot->weak_processor.reset();
                }
            }
            delete processor;
        }
    };

    std::vector<std::shared_ptr<Slot>> slots_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with the number of streams.
 */
template<class T>
LocalProcessorSlots<T>::LocalProcessorSlots(StreamId::size_type num_streams)
{
    CELER_EXPECT(num_streams > 0);
    slots_.reserve(num_streams);
    for ([[maybe_unused]] auto i : range(num_streams))
    {
        slots_.push_back(std::make_shared<Slot>());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get or create the processor for a stream.
 *
 * The \c construct function must return a newly allocated (raw) pointer to
 * the processor. Due to Geant4 multithread semantics, this \b must be called
 * on the same CPU thread on which the resulting processor is used.
 */
template<class T>
template<class F>
auto LocalProcessorSlots<T>::make(StreamId sid, F&& construct) -> SPProcessor
{
    CELER_EXPECT(sid < slots_.size());

    auto const& slot = slots_[sid.get()];
    CELER_ASSERT(slot);
    if (auto result = slot->weak_processor.lock())
    {
        CELER_ASSERT(result.get() == slot->processor);
        return result;
    }

    slot->processor = nullptr;

    SPProcessor result{std::forward<F>(construct)(), SlotDeleter{slot}};
    CELER_ASSERT(result);
    slot->weak_processor = result;
    slot->processor = result.get();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Access the processor for a stream.
 *
 * The processor must have been created with \c make and still be alive.
 */
template<class T>
T& LocalProcessorSlots<T>::get(StreamId sid) const
{
    CELER_EXPECT(sid < slots_.size());
    auto* result = slots_[sid.get()]->processor;
    CELER_EXPECT(result);
    return *result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
