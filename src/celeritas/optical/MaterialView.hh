//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/MaterialView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"

#include "MaterialData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Access optical material properties.
 */
class MaterialView
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<MaterialParamsData>;
    using MaterialId = OpticalMaterialId;
    //!@}

  public:
    // Construct from params and material ID
    inline CELER_FUNCTION MaterialView(ParamsRef const& params, MaterialId id);

    // Construct from params and volume ID
    inline CELER_FUNCTION MaterialView(ParamsRef const& params, VolumeId vol);

    // Whether the view is into an optical material
    inline CELER_FUNCTION operator bool() const;

    //// MATERIAL DATA ////

    // ID of this optical material
    CELER_FORCEINLINE_FUNCTION MaterialId material_id() const;

    // ID of the associated core material
    CELER_FORCEINLINE_FUNCTION CoreMaterialId core_material_id() const;

    //// PARAMETER DATA ////

    // Access energy-dependent refractive index
    inline CELER_FUNCTION NonuniformGridCalculator
    make_refractive_index_calculator() const;

  private:
    //// DATA ////

    ParamsRef const& params_;
    MaterialId mat_id_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from an optical material.
 */
CELER_FUNCTION
MaterialView::MaterialView(ParamsRef const& params, MaterialId id)
    : params_(params), mat_id_(id)
{
    CELER_EXPECT(id < params_.refractive_index.size());
}

//---------------------------------------------------------------------------//
/*!
 * Construct from the current geometry volume.
 */
CELER_FUNCTION
MaterialView::MaterialView(ParamsRef const& params, VolumeId id)
    : params_{params}
{
    CELER_EXPECT(id < params_.optical_id.size());
    mat_id_ = params_.optical_id[id];
}

//---------------------------------------------------------------------------//
/*!
 * Whether the view is into an optical material.
 *
 * This accessor is so that tracks can enter a non-optical region and be killed
 * without crashing the code.
 */
CELER_FORCEINLINE_FUNCTION MaterialView::operator bool() const
{
    return static_cast<bool>(mat_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the optical material id.
 */
CELER_FUNCTION auto MaterialView::material_id() const -> MaterialId
{
    return mat_id_;
}

//---------------------------------------------------------------------------//
/*!
 * Get the id of the core material associated with this optical material.
 */
CELER_FUNCTION CoreMaterialId MaterialView::core_material_id() const
{
    return params_.core_material_id[mat_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Access energy-dependent refractive index.
 */
CELER_FUNCTION NonuniformGridCalculator
MaterialView::make_refractive_index_calculator() const
{
    CELER_EXPECT(*this);
    return NonuniformGridCalculator(params_.refractive_index[mat_id_],
                                    params_.reals);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
