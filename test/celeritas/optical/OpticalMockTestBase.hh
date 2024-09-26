//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalMockTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "../GlobalTestBase.hh"

namespace celeritas
{
class ImportOpticalMaterial;

namespace optical
{
class PhysicsParams;

namespace test
{
using namespace ::celeritas::test;

//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    OpticalMockTestBase ...;
   \endcode
 */
class OpticalMockTestBase : virtual public GlobalTestBase
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstOpticalPhysics = std::shared_ptr<PhysicsParams const>;
    //!@}

  public:
    inline SPConstOpticalPhysics const& optical_physics() const;
    inline SPConstOpticalPhysics const& optical_physics();

  protected:
    SPConstOpticalPhysics build_optical_physics();
    SPConstOpticalMaterial build_optical_material() override;

    SPConstGeo build_geometry() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstMaterial build_material() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstGeoMaterial build_geomaterial() override
    {
        CELER_ASSERT_UNREACHABLE();
    }
    SPConstParticle build_particle() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstCutoff build_cutoff() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstPhysics build_physics() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstSim build_sim() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstTrackInit build_init() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstWentzelOKVI build_wentzel() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction build_along_step() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstCerenkov build_cerenkov() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstScintillation build_scintillation() override
    {
        CELER_ASSERT_UNREACHABLE();
    }

  private:
    SPConstOpticalPhysics optical_physics_;
};

auto OpticalMockTestBase::optical_physics() -> SPConstOpticalPhysics const&
{
    if (!this->optical_physics_)
    {
        this->optical_physics_ = this->build_optical_physics();
        CELER_ASSERT(this->optical_physics_);
    }
    return this->optical_physics_;
}

auto OpticalMockTestBase::optical_physics() const
    -> SPConstOpticalPhysics const&
{
    CELER_ASSERT(this->optical_physics_);
    return this->optical_physics_;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
