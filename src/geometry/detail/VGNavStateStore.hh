//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGNavStateStore.hh
//---------------------------------------------------------------------------//
#ifndef geometry_detail_VGNavStateStore_hh
#define geometry_detail_VGNavStateStore_hh

#include <memory>
#include "base/Span.hh"
#include "base/Types.hh"

#ifdef __NVCC__
#    error "Do not include from CUDA code"
#endif

namespace vecgeom
{
inline namespace cxx
{
class NavStatePool;
} // namespace cxx
} // namespace vecgeom

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Manage a pool of device-side geometry states.
 *
 * Construction of the navstatepool has to be in a host compliation unit due to
 * VecGeom macro magic.
 */
class VGNavStateStore
{
  public:
    // Construct without allocating
    VGNavStateStore() = default;

    // Construct with sizes, allocating on GPU
    VGNavStateStore(size_type size, int depth);

    // View to array of allocated on-device data
    void* device_pointers() const;

  private:
    using NavStatePool = vecgeom::cxx::NavStatePool;

    struct NavStatePoolDeleter
    {
        void operator()(NavStatePool*) const;
    };
    using DeviceUniquePtr = std::unique_ptr<NavStatePool, NavStatePoolDeleter>;

    DeviceUniquePtr pool_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#endif // geometry_detail_VGNavStateStore_hh
