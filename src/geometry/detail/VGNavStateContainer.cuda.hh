//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGNavStateContainer.hh
//---------------------------------------------------------------------------//
#ifndef geometry_detail_VGNavStateContainer_hh
#define geometry_detail_VGNavStateContainer_hh

#include <memory>
#include "VecGeom/base/Cuda.h"
#include "base/Span.hh"
#include "base/Types.hh"

namespace vecgeom
{
VECGEOM_HOST_FORWARD_DECLARE(class NavStatePool;);

inline namespace VECGEOM_IMPL_NAMESPACE
{
class NavStatePool;
} // namespace VECGEOM_IMPL_NAMESPACE
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
class VGNavStateContainer
{
  public:
    // Construct with sizes, allocating on GPU
    VGNavStateContainer(size_type size, int depth);

    //@{
    //! Defaults defined in .cc file
    VGNavStateContainer();
    ~VGNavStateContainer();
    VGNavStateContainer(VGNavStateContainer&&);
    VGNavStateContainer& operator=(VGNavStateContainer&&);
    //@}

    // View to array of allocated on-device data
    void* device_view() const;

  private:
    std::unique_ptr<vecgeom::cxx::NavStatePool> pool_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#endif // geometry_detail_VGNavStateContainer_hh
