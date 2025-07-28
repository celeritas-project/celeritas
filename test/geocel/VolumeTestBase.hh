//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "Test.hh"

namespace celeritas
{
class VolumeParams;

namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Base class for volume tests.
 *
 * This provides common functionality for all volume-based tests.
 */
class VolumeTestBase : public ::celeritas::test::Test
{
  public:
    void SetUp() override;

    // Get the volume parameters
    VolumeParams const& volumes() const;

  protected:
    // Create volume parameters
    virtual std::shared_ptr<VolumeParams> build_volumes() const = 0;

  private:
    std::shared_ptr<VolumeParams> volumes_;
};

//---------------------------------------------------------------------------//
/*!
 * Base for tests with a single volume A.
 */
class SingleVolumeTestBase : public virtual VolumeTestBase
{
  protected:
    std::shared_ptr<VolumeParams> build_volumes() const override;
};

//---------------------------------------------------------------------------//
/*!
 * Base for tests with complex volumes A through E with three
 * instances of C (one inside A, two inside B), placing them in the hierarchy
 * with the following volume instances:
 * \verbatim
   {parent} -> {daughter} "{volume instance label}"
     A -> B "0"
     A -> C "1"
     B -> C "2"
     B -> C "3"
     C -> D "4"
     C -> E "5"
 * \endverbatim
 */
class ComplexVolumeTestBase : public virtual VolumeTestBase
{
  protected:
    std::shared_ptr<VolumeParams> build_volumes() const override;
};

//---------------------------------------------------------------------------//
/*!
 * Base for tests with optical surfaces from `optical-surfaces.gdml`.
 * \verbatim
 * world      -> lar_sphere   "lar_pv"
   world      -> tube2        "tube2_below_pv"
   world      -> tube1_mid    "tube1_mid_pv"
   world      -> tube2        "tube2_above_pv"
 * \endverbatim
 */
class OpticalVolumeTestBase : public virtual VolumeTestBase
{
  protected:
    std::shared_ptr<VolumeParams> build_volumes() const override;
};

//---------------------------------------------------------------------------//
/*!
 * Base for tests with multi-level representation including reflection:
 * \verbatim
   box       -> sph        "boxsph1:0"
   box       -> sph        "boxsph2:0"
   box       -> tri        "boxtri:0"
   world     -> box        "topbox1"
   world     -> sph        "topsph1"
   world     -> box        "topbox2"
   world     -> box        "topbox3"
   world     -> box_refl   "topbox4"
   box_refl  -> sph_refl   "boxsph1:1"
   box_refl  -> sph_refl   "boxsph2:1"
   box_refl  -> tri_refl   "boxtri:1"
 * \endverbatim
 */
class MultiLevelVolumeTestBase : public virtual VolumeTestBase
{
  protected:
    std::shared_ptr<VolumeParams> build_volumes() const override;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
