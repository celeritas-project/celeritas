//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGNavStateStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include "base/Assert.hh"
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
 *
 * This class is designed with a PIMPL-like idiom to hide VecGeom classes from
 * downstream Celeritas code. It specifically also ensures that the
 * construction and destruction of the NavStatePool are compiled using the host
 * compiler (not NVCC), which is necessary by VecGeom's design.
 */
class VGNavStateStore
{
  public:
    //@{
    //! Type aliases
    using NavStatePool = vecgeom::cxx::NavStatePool;
    //@}

  public:
    // Construct without allocating
    VGNavStateStore() = default;

    // Construct with sizes but do not yet copy to GPU
    VGNavStateStore(size_type size, int depth);

    //! Whether the state is constructed
    explicit operator bool() const { return static_cast<bool>(pool_); }

    // Access the host pool (TODO: delete once cuda::GlobalLocator works)
    NavStatePool& get()
    {
        REQUIRE(*this);
        return *pool_;
    }

    // Copy host states to device
    void copy_to_device();

    // View to array of allocated on-device data
    void* device_pointers() const;

  private:
    struct NavStatePoolDeleter
    {
        void operator()(NavStatePool*) const;
    };
    using UPNavStatePool = std::unique_ptr<NavStatePool, NavStatePoolDeleter>;

    UPNavStatePool pool_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
