//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalGenParams.cc
//---------------------------------------------------------------------------//
#include "OpticalGenParams.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
//! Construct a state
template<MemSpace M>
auto make_state(OpticalGenParams const& params, StreamId stream, size_type size)
{
    using StoreT = CollectionStateStore<OpticalGenStateData, M>;

    auto result = std::make_unique<OpticalGenState<M>>();
    result->store = StoreT{params.host_ref(), stream, size};

    CELER_ENSURE(*result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with aux ID and optical data.
 */
OpticalGenParams::OpticalGenParams(AuxId aux_id, OpticalGenSetup const& setup)
    : aux_id_{aux_id}
{
    CELER_EXPECT(aux_id_);
    CELER_EXPECT(setup);

    data_ = CollectionMirror{HostVal<OpticalGenParamsData>{setup}};

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
auto OpticalGenParams::create_state(MemSpace m,
                                    StreamId sid,
                                    size_type size) const -> UPState
{
    if (m == MemSpace::host)
    {
        return make_state<MemSpace::host>(*this, sid, size);
    }
    else if (m == MemSpace::device)
    {
        return make_state<MemSpace::device>(*this, sid, size);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
