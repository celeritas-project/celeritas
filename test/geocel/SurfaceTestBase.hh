//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SurfaceTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "VolumeTestBase.hh"

namespace celeritas
{
class SurfaceParams;
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Base class for surface tests.
 *
 * This provides common functionality for all surface-based tests.
 */
class SurfaceTestBase : public virtual VolumeTestBase
{
  public:
    void SetUp() override;

    //! Get the surface parameters
    SurfaceParams const& surfaces() const;

  protected:
    //! Create surface parameters
    virtual std::shared_ptr<SurfaceParams> build_surfaces() const = 0;

  private:
    std::shared_ptr<SurfaceParams> surfaces_;
};

//---------------------------------------------------------------------------//
/*!
 * Base for tests with many surfaces:
 * \verbatim
    c2b : interface 2 -> 0
    c2c2: interface 2 -> 2
    b   : boundary for A
    cc2 : interface 1 -> 2
    c3c : interface 3 -> 1
    bc  : interface 0 -> 1
    bc2 : interface 0 -> 2
    ec  : interface 5 -> 1
    db  : interface 4 -> 1
 * \endverbatim
 */
class ManySurfacesTestBase : public virtual ComplexVolumeTestBase,
                             public SurfaceTestBase
{
  protected:
    std::shared_ptr<SurfaceParams> build_surfaces() const override;
};

//---------------------------------------------------------------------------//
/*!
 * Base for tests with optical surfaces:
 * \verbatim
    sphere_skin   : boundary for 0 (lar_sphere)
    tube2_skin    : boundary for 1 (tube2)
    below_to_1    : interface 1 -> 2 (tube2_below_pv -> tube1_mid_pv)
    mid_to_below  : interface 2 -> 1 (tube1_mid_pv -> tube2_below_pv)
    mid_to_above  : interface 2 -> 3 (tube1_mid_pv -> tube2_above_pv)
 * \endverbatim
 */
class OpticalSurfacesTestBase : public virtual OpticalVolumeTestBase,
                                public SurfaceTestBase
{
  protected:
    std::shared_ptr<SurfaceParams> build_surfaces() const override;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
