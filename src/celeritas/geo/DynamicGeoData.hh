//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/DynamicGeoData.hh
//! \brief Reusable host-only geometry implementation for dynamic core geometry
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <type_traits>

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "geocel/GeoTrackInterface.hh"
#include "orange/OrangeData.hh"

#if CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_DYNAMIC
#    error "DynamicGeoData.hh requires CELERITAS_CORE_GEO=dynamic"
#endif
#if CELER_USE_DEVICE
#    error "Dynamic core geometry is host-only (incompatible with CUDA/HIP)"
#endif
#if CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK
#    error "Dynamic core geometry is incompatible with OpenMP track threading"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Virtual geometry track implementation used by the dynamic core geometry
using DynamicGeoTrackInterface = GeoTrackInterface<real_type>;

//---------------------------------------------------------------------------//
/*!
 * Owning (value) or borrowed (reference) dynamic geometry implementation.
 *
 * The \c value state data shares ownership of a single heap-allocated
 * implementation that is reused for every track slot (see
 * \c GeoTrackInterface::set_track_slot ). References to the state data hold a
 * raw pointer to it, since state references must be trivially copyable.
 * Because the implementation is shared by all slots of a state, a state must
 * never be accessed by more than one thread at a time.
 */
template<Ownership W>
using DynamicGeoTrackPtr
    = std::conditional_t<W == Ownership::value,
                         std::shared_ptr<DynamicGeoTrackInterface>,
                         DynamicGeoTrackInterface*>;

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
//! Get a non-owning pointer from an owning dynamic geometry implementation
inline DynamicGeoTrackInterface* get_pointer(
    std::shared_ptr<DynamicGeoTrackInterface> const& p)
{
    return p.get();
}

//! Get a non-owning pointer from a borrowed dynamic geometry implementation
inline DynamicGeoTrackInterface* get_pointer(DynamicGeoTrackInterface* p)
{
    return p;
}

// Create a reusable implementation bound to the given params and state
std::shared_ptr<DynamicGeoTrackInterface> make_dynamic_geo_track(
    HostCRef<OrangeParamsData> const& params,
    HostRef<OrangeStateData> const& states);

//---------------------------------------------------------------------------//
}  // namespace celeritas
