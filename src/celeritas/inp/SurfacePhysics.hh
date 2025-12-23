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
#include "corecel/inp/Grid.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
class SurfaceModel;

namespace inp
{
//---------------------------------------------------------------------------//
// SURFACE DESCRIPTION: Reflectivity and models for surface normals.
//---------------------------------------------------------------------------//
/*!
 * Model reflectivity as a user-prescribed function of energy.
 *
 * The grid can also be used to represent a constant reflectivity.
 */
struct GridReflection
{
    //! Reflectivity values [MeV -> unitless]
    Grid reflectivity;

    // Whether the data are assigned
    explicit operator bool() const { return static_cast<bool>(reflectivity); }
};

//---------------------------------------------------------------------------//
/*!
 * Model reflectivity analytically using the Fresnel equations.
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
 * Roughness range is \f$ [0, 1] \f$, where 0 is specular and 1 is diffuse.
 * This parameter is the complement of the \em polish as defined in Geant4:
 * \code roughness = 1 - GetPolish(); \endcode .
 *
 * See \c celeritas::optical::SmearRoughnessSampler .
 *
 * \note This model is used by the GLISUR subroutine in Geant3 and the
 * corresponding "glisur" surface model in Geant4.
 */
struct SmearRoughness
{
    double roughness{-1};  //!< Scale from 0 = specular to 1 = diffuse

    // Whether the data are assigned
    explicit operator bool() const { return roughness >= 0 && roughness <= 1; }
};

//---------------------------------------------------------------------------//
/*!
 * Approximate the microfacet normal distributions as Gaussian.
 *
 * See \c celeritas::optical::GaussianRoughnessSampler .
 *
 * \note This model is used by the "unified" surface model in Geant4.
 */
struct GaussianRoughness
{
    //! Standard deviation of the microfacet slope distribution
    double sigma_alpha{0};

    // Whether the roughness has a meaningful value
    explicit operator bool() const { return sigma_alpha > 0; }
};

//---------------------------------------------------------------------------//
// SURFACE PHYSICS: interaction mechanisms / reflection models.
//---------------------------------------------------------------------------//
/*!
 * Parameterization of the UNIFIED reflection model.
 *
 * The reflection grids store the probability of each angular distribution in
 * the UNIFIED model:
 * - \c specular_spike : Reflection probability at the average surface normal.
 * - \c specular_lobe : Reflection probability at the micro facet normal.
 * - \c backscatter : Probability of backscattering after reflecting within a
 *   deep groove.
 *
 * The sum of all three parameters must be <= 1 at every grid point, with the
 * remainder being the probability of diffuse scattering.
 *
 * \todo We could require these to all be on the same energy grid for improved
 * performance and error checking.
 */
struct ReflectionForm
{
    //!@{
    //! \name Type aliases
    using Mode = optical::ReflectionMode;
    using ReflectionGrids = EnumArray<Mode, Grid>;
    //!@}

    //! Probability of reflection for each reflection mode
    ReflectionGrids reflection_grids;

    // Whether all grids are specified
    explicit operator bool() const
    {
        return std::all_of(
            reflection_grids.begin(),
            reflection_grids.end(),
            [](auto const& grid) { return static_cast<bool>(grid); });
    }

    //! Return a specular spike reflection form
    static ReflectionForm from_spike()
    {
        return ReflectionForm::from_mode(Mode::specular_spike);
    }

    //! Return a specular lobe reflection form
    static ReflectionForm from_lobe()
    {
        return ReflectionForm::from_mode(Mode::specular_lobe);
    }

    //! Return a Lambertian (diffuse) reflection form
    static ReflectionForm from_lambertian()
    {
        return ReflectionForm::from_mode(Mode::diffuse_lobe);
    }

    //! Construct a reflection form with only one active grid
    static ReflectionForm from_mode(Mode only_mode)
    {
        ReflectionForm result;
        for (auto mode : range(optical::ReflectionMode::size_))
        {
            result.reflection_grids[mode]
                = Grid::from_constant(mode == only_mode ? 1 : 0);
        }
        return result;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Analytic interactions between dielectric and dielectric or metal materials.
 */
struct DielectricInteraction
{
    //! Exiting angular distributions
    ReflectionForm reflection;

    //! Whether the interface is dielectric-dielectric or dielectric-metal
    bool is_metal{false};

    //! Return a dielectric-dielectric interaction
    static DielectricInteraction from_dielectric(ReflectionForm r)
    {
        return {std::move(r), false};
    }

    //! Return a dielectric-metal interaction
    static DielectricInteraction from_metal(ReflectionForm r)
    {
        return {std::move(r), true};
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
    //! Perfectly smooth surfaces
    std::map<PhysSurfaceId, NoRoughness> polished;
    //! Surfaces using the "smear" roughness model
    std::map<PhysSurfaceId, SmearRoughness> smear;
    //! Surfaces using the "gaussian" roughness model
    std::map<PhysSurfaceId, GaussianRoughness> gaussian;

    // Whether any models are present
    explicit operator bool() const
    {
        return !polished.empty() || !smear.empty() || !gaussian.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Reflectivity mechanism.
 */
struct ReflectivityModels
{
    //! Surfaces using an energy-dependent, user-specified grid
    std::map<PhysSurfaceId, GridReflection> grid;
    //! Surfaces using the analytic Fresnel equations
    std::map<PhysSurfaceId, FresnelReflection> fresnel;

    // Whether any models are present
    explicit operator bool() const
    {
        return !grid.empty() || !fresnel.empty();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Interaction models for different interface types.
 *
 * This will be extended to allow user-provided interaction kernels.
 */
struct InteractionModels
{
    //! Composite reflection distributions at a dielectric interface
    std::map<PhysSurfaceId, DielectricInteraction> dielectric;

    //! Trivial interactions independent of other surface physics
    std::map<PhysSurfaceId, optical::TrivialInteractionMode> trivial;

    // Whether any models are present
    explicit operator bool() const
    {
        return !dielectric.empty() || !trivial.empty();
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
 *
 * \todo rename OpticalSurfacePhysics
 */
struct SurfacePhysics
{
    //!@{
    //! \name type aliases
    using DetectionEfficiency = std::map<PhysSurfaceId, Grid>;
    using VecInterstitialMaterials = std::vector<OptMatId>;
    //!@}

    std::vector<VecInterstitialMaterials> materials;

    //! Microfacet distribution models
    RoughnessModels roughness;
    //! Reflectivity models
    ReflectivityModels reflectivity;
    //! Reflection+refraction+absorption models
    InteractionModels interaction;

    // Whether the data are assigned
    explicit operator bool() const
    {
        return roughness && reflectivity && interaction && !materials.empty();
    }
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
