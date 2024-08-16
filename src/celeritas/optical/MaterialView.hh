//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/MaterialView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericCalculator.hh"

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

    //// MATERIAL DATA ////

    // ID of this optical material
    CELER_FORCEINLINE_FUNCTION MaterialId material_id() const;

    //// PARAMETER DATA ////

    // Access energy-dependent refractive index
    GenericCalculator make_refractive_index_calculator() const;

  private:
    //// DATA ////

    ParamsRef const& params_;
    MaterialId mat_id_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from dynamic and static particle properties.
 */
CELER_FUNCTION
MaterialView::MaterialView(ParamsRef const& params, MaterialId id)
    : params_(params), mat_id_(id)
{
    CELER_EXPECT(id < params_.refractive_index.size());
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
 * Access energy-dependent refractive index.
 */
CELER_FUNCTION GenericCalculator
MaterialView::make_refractive_index_calculator() const
{
    return GenericCalculator(params_.refractive_index[mat_id_], params_.reals);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
