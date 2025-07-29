//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/SurfacePhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <map>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/Types.hh"
#include "celeritas/inp/Grid.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
// SURFACE DESCRIPTION: Reflectivity and models for surface normals.
//---------------------------------------------------------------------------//
/*!
 * Surface reflectivity index, which can be a user-defined Grid, which is a
 * function of energy (wavelength), or constant.
 */
struct ReflectionGrid
{
    Grid grid;

    //! Whether the data are assigned
    explicit operator bool() const { return static_cast<bool>(grid); }
};

//---------------------------------------------------------------------------//
/*!
 * Analytic reflectivity: use Fresnel equations.
 */
struct ReflectionAnalytic
{
};

//---------------------------------------------------------------------------//
/*!
 * Simplest surface normal treatment.
 */
struct Polished
{
};

//---------------------------------------------------------------------------//
/*!
 * Global surface normal with smearing.
 *
 * Roughness range is [0, 1], where 1 is maximum roughness. In Geant4 parlance,
 * \code roughness = 1 - GetPolish(); \endcode.
 *
 * \note Used by the GLISUR model in Geant4.
 */
struct SmearRoughness
{
    real_type roughness{-1};  //!< [0, 1] where 1 means maximum roughness

    //! Whether the data are assigned
    explicit operator bool() const { return roughness >= 0 && roughness <= 1; }
};

//---------------------------------------------------------------------------//
/*!
 * Assumes a Gaussian distribution of the facet normal with a \f$ \sigma_\alpha
 * \f$ standard deviation.
 *
 * \note Used by the Unified model in Geant4.
 */
struct GaussianRoughness
{
    real_type sigma_alpha{-1};  //!< Gaussian std. dev.

    //! Whether the surface data are assigned
    explicit operator bool() const { return sigma_alpha > 0; }
};

//---------------------------------------------------------------------------//
//!@{
//! \name Convenience typedef for current simplified layer implementation.
//! \todo: Expand \c SurfaceLayer to a `map<SurfaceId, SurfaceLayerId>`, where
//! the \c SurfaceLayerId describes a set of layers.
using SurfaceLayer = SurfaceId;
//!@}

//---------------------------------------------------------------------------//
// SURFACE PHYSICS: interaction mechanisms / reflection models.
//---------------------------------------------------------------------------//
/*!
 * Paramaters used by different reflection mechanisms.
 *
 * Parameters:
 * - \todo: Add \c lambertian_roughness : Roughness parameter used by
 *   Lambertian reflection.
 * - \c specular_lobe : Reflection probability at the micro facet normal.
 * - \c specular_spike : Reflection probability at the average surface normal.
 * - \c backscatter : Probability of back scattering after reflecting within a
 *   deep groove.
 * - \c diffuse_lobe : Probability of internal Lambertian reflection.
 *
 * \note The sum of \c specular_lobe + \c specular_spike + \c backscatter +
 * \c diffuse_lobe probabilities must be equal to one. Diffuse lobe is not
 * user-defined in Geant4 and is calculated from the other three. This
 * verification must be done during import (filling \c inp:: data) and/or
 * construction of Celeritas' params data.
 */
struct ReflectionForm
{
    // The sum of these properties must be equal to 1
    Grid specular_lobe;  //!< [0, 1] probability
    Grid specular_spike;  //!< [0, 1] probability
    Grid backscatter;  //!< [0, 1] probability
    Grid diffuse_lobe;  //!< [0, 1] probability (optional)

    //! Whether the data are assigned
    explicit operator bool() const
    {
        return specular_lobe && specular_spike && backscatter;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Interaction models for different interface types.
 *
 * Existing interface types are dielectrict-dielectric and dielectric-metal.
 *
 * \todo Future work may allow for custom interfaces.
 */
struct InteractionModels
{
    std::map<SurfaceLayer, ReflectionForm> dielectric_dielectric;
    std::map<SurfaceLayer, ReflectionForm> dielectric_metal;

    //! Whether the data are assigned
    explicit operator bool() const
    {
        return !dielectric_dielectric.empty() || !dielectric_metal.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Surface roughness description.
 *
 * \todo Future work will allow the of use multiple surface paints/wrappings
 * managed by different models. \c SurfaceLayer will pair a \c SurfaceId with a
 * \c SurfaceLayerId that defiens paint/wrapping combinations.
 */
struct RoughnessModels
{
    std::map<SurfaceLayer, Polished> polished;
    std::map<SurfaceLayer, SmearRoughness> smear;
    std::map<SurfaceLayer, GaussianRoughness> gaussian;

    //! Whether the data are assigned
    explicit operator bool() const
    {
        return !polished.empty() || !smear.empty() || !gaussian.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Reflectivity mechanism.
 *
 * Can be user-defined (grid) and/or analytic (Fresnel equations).
 */
struct ReflectivityModels
{
    std::map<SurfaceLayer, ReflectionGrid> grid;
    std::map<SurfaceLayer, ReflectionAnalytic> analytic;

    //! Whether the data are assigned
    explicit operator bool() const
    {
        return !grid.empty() || !analytic.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Surface physics definition.
 *
 * Maps all optical surfaces with interaction models and surface parameters.
 */
struct SurfacePhysics
{
    //!@{
    //! \name type aliases
    using DetectionEfficiency = std::map<SurfaceLayer, Grid>;
    //!@}

    ReflectivityModels reflectivity;
    RoughnessModels roughness;
    InteractionModels interaction;

    //! Whether the data are assigned
    explicit operator bool() const
    {
        return reflectivity && roughness && interaction;
    }
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
