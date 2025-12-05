//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/DirectGeneratorData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/optical/TrackInitializer.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Direct optical initialization data.
 */
template<Ownership W, MemSpace M>
struct DirectGeneratorStateData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, W, M>;
    //!@}

    // Buffer of track initializers to generate
    Items<TrackInitializer> initializers;

    //! State size
    CELER_FUNCTION size_type size() const { return initializers.size(); }

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !initializers.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    DirectGeneratorStateData<W, M>&
    operator=(DirectGeneratorStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        initializers = other.initializers;
        CELER_ENSURE(*this);
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store direct optical generation states in aux data.
 */
template<MemSpace M>
struct DirectGeneratorState : public GeneratorStateBase
{
    CollectionStateStore<DirectGeneratorStateData, M> store;

    //! Access valid range of track initializers
    auto initializers()
    {
        return this->store.ref().initializers[ItemRange<TrackInitializer>(
            ItemId<TrackInitializer>(this->counters.buffer_size))];
    }

    //! True if states have been allocated
    explicit operator bool() const { return static_cast<bool>(store); }
};

//---------------------------------------------------------------------------//
/*!
 * Resize optical buffers.
 */
template<MemSpace M>
void resize(DirectGeneratorStateData<Ownership::value, M>* state,
            StreamId,
            size_type size)
{
    CELER_EXPECT(size > 0);
    resize(&state->initializers, size);
    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
