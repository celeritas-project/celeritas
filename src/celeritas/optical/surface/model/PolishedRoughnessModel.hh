//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/PolishedRoughnessModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "celeritas/inp/SurfacePhysics.hh"
#include "celeritas/optical/surface/SurfaceModel.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    PolishedRoughnessModel ...;
   \endcode
 */
class PolishedRoughnessModel : public SurfaceModel
{
  public:
    //!@{
    //! \name Type aliases
    //!@}

  public:
    PolishedRoughnessModel(SurfaceModelId,
                           std::map<PhysSurfaceId, inp::NoRoughness> const&);
    std::vector<PhysSurfaceId> get_surfaces() const final;
    void step(CoreParams const&, CoreStateHost&) const final;
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    std::vector<PhysSurfaceId> surfaces_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
