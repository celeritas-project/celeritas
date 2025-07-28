//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Surfaces.hh
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
 * Polishness range is [0, 1], where 1 is maximum polishness.
 *
 * \note Used by the GLISUR model in Geant4.
 */
struct SmearRoughness
{
    real_type polishness{-1};  //!< [0, 1] where 1 means maximum polishness

    //! Whether the data are assigned
    explicit operator bool() const
    {
        return polishness >= 0 && polishness <= 1;
    }
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

//---------------------------------------------------------------------------//
/*!
 * Paramaters used by different reflection mechanisms.
 *
 * Parameters:
 * - \todo: Add \c lambertian_roughness : Roughness parameter used by
 *   Lambertian reflection.
 * - \c specular_lobe : Reflection probability at the micro facet normal.
 * - \c specular_spike : Reflection probability at the average surface normal.
 * - \c back_scatter : Probability of back scattering after reflecting within a
 *   deep groove.
 * - \c diffuse_lobe : Probability of internal Lambertian reflection.
 *
 * \note The sum of \c specular_lobe + \c specular_spike + \c back_scatter +
 * \c diffuse_lobe probabilities must be equal to one. Diffuse lobe is not
 * user-defined in Geant4 and is calculated from the other three.
 */
struct ReflectionForm
{
    // The sum of these properties must be equal to 1
    Grid specular_lobe;  //!< [0, 1] probability
    Grid specular_spike;  //!< [0, 1] probability
    Grid back_scatter;  //!< [0, 1] probability
    Grid diffuse_lobe;  //!< [0, 1] probability

    bool unity(Grid const& grid) const
    {
        return std::any_of(
            grid.y.begin(), grid.y.end(), [](real_type const& val) {
                return val >= 0 && val <= 1;
            });
    }

    // Total probability for all 4 properties must be equal to 1
    bool total_prob_is_unity() const
    {
        auto const& sl = this->specular_lobe;
        auto const& ss = this->specular_spike;
        auto const& bc = this->back_scatter;
        auto const& dl = this->diffuse_lobe;
        auto const size = sl.x.size();
        CELER_ASSERT(ss.x.size() == size && bc.x.size() == size
                     && dl.x.size() == size);

        auto prob_sum = [&](size_type index) -> real_type {
            return sl.y[index] + ss.y[index] + bc.y[index] + dl.y[index];
        };

        for (auto i : range(sl.x.size()))
        {
            if (!soft_equal(real_type{1}, prob_sum(i)))
            {
                return false;
            }
        }
        return true;
    }

    //! Whether the data are assigned
    explicit operator bool() const
    {
        return unity(specular_lobe) && unity(specular_spike)
               && unity(back_scatter) && unity(diffuse_lobe)
               && total_prob_is_unity();
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
        return !polished.empty() || !polished.empty() || !gaussian.empty();
    }
};

//---------------------------------------------------------------------------//
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
    using SurfaceNames = std::map<SurfaceLayer, std::string>;
    using DetectionEfficiency = std::map<SurfaceLayer, Grid>;
    //!@}

    SurfaceNames names;
    ReflectivityModels reflectivity;
    RoughnessModels roughness;
    InteractionModels interaction;
    DetectionEfficiency efficiency;  //!< \todo: Keep as optional?

    //! Whether the data are assigned
    explicit operator bool() const
    {
        return !names.empty() && reflectivity && roughness && interaction;
    }
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
