//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/SurfacePhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <map>

#include "corecel/cont/Range.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/inp/Grid.hh"

namespace celeritas
{
class SurfaceModel;

namespace inp
{
//---------------------------------------------------------------------------//
// SURFACE DESCRIPTION: Reflectivity and models for surface normals.
//---------------------------------------------------------------------------//
/*!
 * Energy-dependent surface reflectivity.
 *
 * The grid can also be used to represent a constant reflectivity.
 */
struct GridReflection
{
    Grid reflectivity;

    //! Whether the data are assigned
    explicit operator bool() const { return static_cast<bool>(reflectivity); }
};

//---------------------------------------------------------------------------//
/*!
 * Analytic reflectivity using the Fresnel equations.
 */
struct FresnelReflection
{
};

//---------------------------------------------------------------------------//
/*!
 * A polished (perfectly smooth) surface.
 *
 * For smooth surfaces, the facet normal is the macroscopic normal.
 */
struct NoRoughness
{
};

//---------------------------------------------------------------------------//
/*!
 * Global surface normal with smearing.
 *
 * Roughness range is [0, 1], where 0 is specular and 1 is diffuse. This
 * parameter is also the complement of the one defined in Geant4:
 * \code roughness = 1 - GetPolish(); \endcode .
 *
 * \note Used by the GLISUR model in Geant4.
 */
struct SmearRoughness
{
    double roughness{-1};  //!< Scale from 0 = specular to 1 = diffuse

    //! Whether the data are assigned
    explicit operator bool() const { return roughness >= 0 && roughness <= 1; }
};

//---------------------------------------------------------------------------//
/*!
 * Approximate the microfacet normal distributions as Gaussian.
 *
 * This is used by the UNIFIED model in Geant4 \citep{levin-morephysical-1996,
 * http://ieeexplore.ieee.org/document/591410/} : the mean of the distribution
 * is zero (as a modification to the macroscopic normal) and the standard
 * deviation is \c sigma_alpha .
 */
struct GaussianRoughness
{
    double sigma_alpha{-1};  //!< Gaussian std. dev.

    //! Whether the surface data are assigned
    explicit operator bool() const { return sigma_alpha > 0; }
};

//---------------------------------------------------------------------------//
// SURFACE PHYSICS: interaction mechanisms / reflection models.
//---------------------------------------------------------------------------//
/*!
 * Parameterization of the UNIFIED reflection model.
 *
 * Parameters:
 * - \c specular_spike : Reflection probability at the average surface normal.
 * - \c specular_lobe : Reflection probability at the micro facet normal.
 * - \c backscatter : Probability of back scattering after reflecting within a
 *   deep groove.
 *
 * The sum of all three parameters must be < 1 at every grid point, with the
 * remainder being the probability of diffuse scattering.
 */
struct ReflectionForm
{
    Grid specular_spike;  //!< [0, 1] probability
    Grid specular_lobe;  //!< [0, 1] probability
    Grid backscatter;  //!< [0, 1] probability

    //! Whether the data are assigned
    explicit operator bool() const
    {
        return specular_lobe && specular_spike && backscatter;
    }

    //! Return a specular spike reflection form
    static ReflectionForm from_spike()
    {
        ReflectionForm result;
        result.specular_spike = Grid::from_constant(1.0);
        result.specular_lobe = Grid::from_constant(0);
        result.backscatter = Grid::from_constant(0);
        return result;
    }

    //! Return a specular lobe reflection form
    static ReflectionForm from_lobe()
    {
        ReflectionForm result;
        result.specular_spike = Grid::from_constant(0);
        result.specular_lobe = Grid::from_constant(1.0);
        result.backscatter = Grid::from_constant(0);
        return result;
    }

    //! Return a Lambertian reflection form
    static ReflectionForm from_lambertian()
    {
        ReflectionForm result;
        result.specular_spike = Grid::from_constant(0);
        result.specular_lobe = Grid::from_constant(0);
        result.backscatter = Grid::from_constant(0);
        return result;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Surface roughness description.
 *
 * \todo Future work will allow the of use multiple surface
 * paints/wrappings managed by different models. \c PhysSurfaceId will pair
 * a \c SurfaceId with a \c PhysSurfaceId that defines paint/wrapping
 * combinations.
 */
struct RoughnessModels
{
    std::map<PhysSurfaceId, NoRoughness> polished;
    std::map<PhysSurfaceId, SmearRoughness> smear;
    std::map<PhysSurfaceId, GaussianRoughness> gaussian;

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
    std::map<PhysSurfaceId, GridReflection> grid;
    std::map<PhysSurfaceId, FresnelReflection> fresnel;

    //! Whether the data are assigned
    explicit operator bool() const
    {
        return !grid.empty() || !fresnel.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Interaction models for different interface types.
 *
 * Existing interface types are dielectric-dielectric and dielectric-metal.
 *
 * This will be extended to allow user-provided interaction kernels.
 */
struct InteractionModels
{
    std::map<PhysSurfaceId, ReflectionForm> dielectric_dielectric;
    std::map<PhysSurfaceId, ReflectionForm> dielectric_metal;

    //! Whether the data are assigned
    explicit operator bool() const
    {
        return !dielectric_dielectric.empty() || !dielectric_metal.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Surface physics definition.
 *
 * Maps all optical surfaces with interaction models and surface
 * parameters.
 *
 * Interstitial materials are the interstitial materials per geometric surface.
 * The last entry is used as the default surface.
 */
struct SurfacePhysics
{
    //!@{
    //! \name type aliases
    using DetectionEfficiency = std::map<PhysSurfaceId, Grid>;
    using VecInterstitialMaterials = std::vector<OptMatId>;
    //!@}

    std::vector<VecInterstitialMaterials> materials;

    RoughnessModels roughness;
    ReflectivityModels reflectivity;
    InteractionModels interaction;

    //! Whether the data are assigned
    explicit operator bool() const
    {
        return reflectivity && roughness && interaction && !materials.empty();
    }
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
