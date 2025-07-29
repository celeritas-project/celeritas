//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSurfacePhysicsLoader.cc
//---------------------------------------------------------------------------//
#include "GeantSurfacePhysicsLoader.hh"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <G4LogicalSurface.hh>
#include <G4OpticalSurface.hh>
#include <G4Version.hh>

#include "corecel/io/Logger.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/SurfaceParams.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the \c G4SurfaceType selection.
 */
char const* to_cstring(G4SurfaceType value)
{
#define GSPL_ST_PAIR(ENUMVALUE)              \
    {                                        \
        G4SurfaceType::ENUMVALUE, #ENUMVALUE \
    }

    static std::unordered_map<G4SurfaceType, std::string> const to_cstring_impl
        = {GSPL_ST_PAIR(dielectric_metal),
           GSPL_ST_PAIR(dielectric_dielectric),
           GSPL_ST_PAIR(dielectric_LUT),
           GSPL_ST_PAIR(dielectric_LUTDAVIS),
           GSPL_ST_PAIR(dielectric_dichroic),
           GSPL_ST_PAIR(firsov),
           GSPL_ST_PAIR(x_ray),
#if G4VERSION_NUMBER >= 1110
           GSPL_ST_PAIR(coated)
#endif
        };
    return to_cstring_impl.find(value)->second.c_str();

#undef GSPL_ST_PAIR
}
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the \c G4OpticalSurfaceModel selection.
 */
char const* to_cstring(G4OpticalSurfaceModel value)
{
#define GSPL_OSM_PAIR(ENUMVALUE)                     \
    {                                                \
        G4OpticalSurfaceModel::ENUMVALUE, #ENUMVALUE \
    }

    static std::unordered_map<G4OpticalSurfaceModel, std::string> const to_cstring_impl
        = {GSPL_OSM_PAIR(glisur),
           GSPL_OSM_PAIR(unified),
           GSPL_OSM_PAIR(LUT),
           GSPL_OSM_PAIR(DAVIS),
           GSPL_OSM_PAIR(dichroic)};
    return to_cstring_impl.find(value)->second.c_str();

#undef GSPL_OSM_PAIR
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the \c G4OpticalSurfaceFinish selection.
 */
char const* to_cstring(G4OpticalSurfaceFinish value)
{
#define GSPL_OSF_PAIR(ENUMVALUE)                      \
    {                                                 \
        G4OpticalSurfaceFinish::ENUMVALUE, #ENUMVALUE \
    }

    static std::unordered_map<G4OpticalSurfaceFinish, std::string> const
        to_cstring_impl
        = {
            GSPL_OSF_PAIR(polished),
            GSPL_OSF_PAIR(polishedfrontpainted),
            GSPL_OSF_PAIR(polishedbackpainted),

            GSPL_OSF_PAIR(ground),
            GSPL_OSF_PAIR(groundfrontpainted),
            GSPL_OSF_PAIR(groundbackpainted),

            GSPL_OSF_PAIR(polishedlumirrorair),
            GSPL_OSF_PAIR(polishedlumirrorglue),
            GSPL_OSF_PAIR(polishedair),
            GSPL_OSF_PAIR(polishedteflonair),
            GSPL_OSF_PAIR(polishedtioair),
            GSPL_OSF_PAIR(polishedtyvekair),
            GSPL_OSF_PAIR(polishedvm2000air),
            GSPL_OSF_PAIR(polishedvm2000glue),

            GSPL_OSF_PAIR(etchedlumirrorair),
            GSPL_OSF_PAIR(etchedlumirrorglue),
            GSPL_OSF_PAIR(etchedair),
            GSPL_OSF_PAIR(etchedteflonair),
            GSPL_OSF_PAIR(etchedtioair),
            GSPL_OSF_PAIR(etchedtyvekair),
            GSPL_OSF_PAIR(etchedvm2000air),
            GSPL_OSF_PAIR(etchedvm2000glue),

            GSPL_OSF_PAIR(groundlumirrorair),
            GSPL_OSF_PAIR(groundlumirrorglue),
            GSPL_OSF_PAIR(groundair),
            GSPL_OSF_PAIR(groundteflonair),
            GSPL_OSF_PAIR(groundtioair),
            GSPL_OSF_PAIR(groundtyvekair),
            GSPL_OSF_PAIR(groundvm2000air),
            GSPL_OSF_PAIR(groundvm2000glue),

            GSPL_OSF_PAIR(Rough_LUT),
            GSPL_OSF_PAIR(RoughTeflon_LUT),
            GSPL_OSF_PAIR(RoughESR_LUT),
            GSPL_OSF_PAIR(RoughESRGrease_LUT),

            GSPL_OSF_PAIR(Polished_LUT),
            GSPL_OSF_PAIR(PolishedTeflon_LUT),
            GSPL_OSF_PAIR(PolishedESR_LUT),
            GSPL_OSF_PAIR(PolishedESRGrease_LUT),

            GSPL_OSF_PAIR(Detector_LUT),
        };
    return to_cstring_impl.find(value)->second.c_str();

#undef GSPL_OSF_PAIR
}

//---------------------------------------------------------------------------//
/*!
 * Verify that all elements of a grid are within range [0, 1].
 *
 * Used to verify that \c ReflectionForm Grids are within the expected range.
 */
bool unity(inp::Grid const& grid)
{
    return std::any_of(grid.y.begin(), grid.y.end(), [](real_type const& val) {
        return val >= 0 && val <= 1;
    });
}

//---------------------------------------------------------------------------//
/*!
 * Verify that the sum of all \c ReflectionForm probabilities adds up to one.
 */
bool total_prob_is_unity(inp::ReflectionForm const& rf)
{
    auto const& sl = rf.specular_lobe;
    auto const& ss = rf.specular_spike;
    auto const& bc = rf.backscatter;
    auto const& dl = rf.diffuse_lobe;
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

//---------------------------------------------------------------------------//
/*!
 * Calculate diffuse lobe from the rest of the imported data.
 *
 * Since the total probability for all 4 properties is equal to one, the
 * diffuse lobe can be calculated by subtracting the other three.
 */
inp::Grid calc_diffuse_lobe(inp::ReflectionForm const& refl_form)
{
    auto const& sl = refl_form.specular_lobe;
    auto const& ss = refl_form.specular_spike;
    auto const& bc = refl_form.backscatter;
    auto const size = sl.x.size();
    CELER_ASSERT(ss.x.size() == size && bc.x.size() == size);

    inp::Grid result;
    result.x = sl.x;
    result.y.resize(size);
    for (auto i : range(size))
    {
        // diffuse_lobe = 1 - specular_lobe - specular_spike - backscatter
        result.y[i] = real_type{1} - sl.y[i] - ss.y[i] - bc.y[i];
    }
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
GeantSurfacePhysicsLoader::GeantSurfacePhysicsLoader() {}

//---------------------------------------------------------------------------//
/*!
 * Populate surface physics data for a given \c SurfaceId .
 */
void GeantSurfacePhysicsLoader::operator()(SurfaceId sid,
                                           inp::SurfacePhysics& result)
{
    CELER_EXPECT(sid);
    detail::GeantSurfacePhysicsHelper helper(sid);
    try
    {
        this->insert_reflectivity(helper, result);
        this->insert_roughness(helper, result);
        this->insert_interaction(helper, result);

        // Ensure that data is compatible with selected model
        this->validate_model(helper, result);
    }
    catch (RuntimeError const& e)
    {
        auto const& surf = helper.surface();
        CELER_LOG(error) << "failed to convert surface " << surf.GetName()
                         << " (id " << sid.unchecked_get() << ") with model "
                         << to_cstring(surf.GetModel()) << " and finish "
                         << to_cstring(surf.GetFinish()) << ": "
                         << e.details().which << ", " << e.details().what;
    }
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Collect reflectivity information from a given optical surface.
 */
void GeantSurfacePhysicsLoader::insert_reflectivity(
    detail::GeantSurfacePhysicsHelper& helper, inp::SurfacePhysics& result)
{
    auto sid = helper.surface_id();
    inp::ReflectivityModels refl_mods;
    if (!this->analytic_reflection_only(helper.surface()))
    {
        // Insert any model that includes user-defined grid reflectivity
        inp::ReflectionGrid refl_grid;
        helper.get_property(&refl_grid.grid, "REFLECTIVITY");
        refl_mods.grid.insert({sid, std::move(refl_grid)});
    }
    refl_mods.analytic.insert({sid, inp::ReflectionAnalytic{}});
    result.reflectivity = std::move(refl_mods);
    CELER_ENSURE(result.reflectivity);
}

//---------------------------------------------------------------------------//
/*!
 * Collect roughness information from a given optical surface.
 */
void GeantSurfacePhysicsLoader::insert_roughness(
    detail::GeantSurfacePhysicsHelper& helper, inp::SurfacePhysics& result)
{
    using G4OSM = G4OpticalSurfaceModel;

    auto sid = helper.surface_id();
    auto const& surf = helper.surface();
    auto const g4model = surf.GetModel();
    switch (g4model)
    {
        case G4OSM::glisur: {
            // Get GLISUR surface polish
            real_type roughness = real_type{1} - surf.GetPolish();
            if (roughness == soft_equal(real_type{0}, roughness))
            {
                // Perfectly polished surface
                result.roughness.polished.insert({sid, inp::Polished{}});
            }
            else
            {
                // Smearing is available
                inp::SmearRoughness smear{roughness};
                CELER_ASSERT(smear);
                result.roughness.smear.insert({sid, std::move(smear)});
            }
            break;
        }

        case G4OSM::unified: {
            // Insert Gaussian if available
            inp::GaussianRoughness gauss;
            gauss.sigma_alpha = surf.GetSigmaAlpha();
            if (gauss)
            {
                result.roughness.gaussian.insert({sid, std::move(gauss)});
            }
            break;
        }

        default:
            CELER_LOG(error) << "G4OpticalSurfaceModel '"
                             << to_cstring(g4model) << "' not available";
            break;
    }
    CELER_ENSURE(result.roughness);
}

//---------------------------------------------------------------------------//
/*!
 * Collect interaction information from a given optical surface.
 */
void GeantSurfacePhysicsLoader::insert_interaction(
    detail::GeantSurfacePhysicsHelper& helper, inp::SurfacePhysics& result)
{
    using G4ST = G4SurfaceType;

#define GSPL_VALIDATE_UNITY(PARAM)                           \
    CELER_VALIDATE(unity(PARAM),                             \
                   << "ReflectionForm parameter '" << #PARAM \
                   << "' is not within [0, 1] range")

    inp::ReflectionForm refl_form;
    helper.get_property(&refl_form.specular_lobe, "SPECULARLOBECONSTANT");
    helper.get_property(&refl_form.specular_spike, "SPECULARSPIKECONSTANT");
    helper.get_property(&refl_form.backscatter, "BACKSCATTERCONSTANT");
    refl_form.diffuse_lobe = calc_diffuse_lobe(refl_form);
    CELER_ASSERT(refl_form);

    // Verify unity and total probability of reflection form parameters
    GSPL_VALIDATE_UNITY(refl_form.specular_spike);
    GSPL_VALIDATE_UNITY(refl_form.specular_lobe);
    GSPL_VALIDATE_UNITY(refl_form.backscatter);
    GSPL_VALIDATE_UNITY(refl_form.diffuse_lobe);
    CELER_VALIDATE(total_prob_is_unity(refl_form),
                   << "The sum of all ReflectionForm probabilities is "
                      "not equal to 1");

    // ReflectionForm terms are correctly assigned; Add to interface type
    auto sid = helper.surface_id();
    auto const interface_type = helper.surface().GetType();
    switch (interface_type)
    {
        case G4ST::dielectric_dielectric:
            result.interaction.dielectric_dielectric.insert(
                {sid, std::move(refl_form)});
            break;
        case G4ST::dielectric_metal:
            result.interaction.dielectric_metal.insert(
                {sid, std::move(refl_form)});
            break;
        default:
            CELER_LOG(error) << "G4SurfaceType '" << to_cstring(interface_type)
                             << "' not available";
            break;
    }

#undef GSPL_VALIDATE_UNITY

    CELER_ENSURE(result.interaction);
}

//---------------------------------------------------------------------------//
/*!
 * Return true for Geant4 models/finishes that *ONLY* use analytical
 * reflection.
 *
 * Currently, only the Unified model with [polished/ground]backpainted undergo
 * uniquely through analytical reflection (i.e. Fresnel equations).
 */
bool GeantSurfacePhysicsLoader::analytic_reflection_only(
    G4OpticalSurface const& surf) const
{
    using G4OSM = G4OpticalSurfaceModel;
    using G4OSF = G4OpticalSurfaceFinish;

    if (surf.GetModel() == G4OSM::unified)
    {
        if (surf.GetFinish() == G4OSF::polishedbackpainted
            || surf.GetFinish() == G4OSF::groundbackpainted)
        {
            // Unified [polished/ground]backpainted are *only* analytic
            return true;
        }
    }

    return false;
}

//---------------------------------------------------------------------------//
/*!
 * Ensure that a mapped optical surface does not have inconsistent model data
 * assigned to it.
 *
 * Minimum requirements for each implemented model:
 * - GLISUR
 *   - Roughness: uses polished or smear; Gaussian is never used.
 * - Unified
 *   - Roughness: uses Gaussian or polished; smear is never used.
 *   - ReflectiomForm: \c specular_spike , \c specular_lobe , \c backscatter .
 */
void GeantSurfacePhysicsLoader::validate_model(
    detail::GeantSurfacePhysicsHelper& helper, inp::SurfacePhysics& result) const
{
    using G4OSM = G4OpticalSurfaceModel;

#define GSPL_IS_MAPPED(MEMBER) (result.MEMBER.find(sid) != result.MEMBER.end())

    auto sid = helper.surface_id();
    auto const& surf = helper.surface();
    auto const model = surf.GetModel();
    switch (model)
    {
        case G4OSM::glisur:
            // Minimum required data
            CELER_VALIDATE((GSPL_IS_MAPPED(roughness.polished)
                            || GSPL_IS_MAPPED(roughness.smear)),
                           << "Missing polished or smear surface for the "
                              "GLISUR model");

            // Expected empty maps
            CELER_VALIDATE(!GSPL_IS_MAPPED(roughness.gaussian),
                           << "Gaussian surface cannot be added to GLISUR "
                              "model");
            break;

        case G4OSM::unified:
            // Minimum required data
            CELER_VALIDATE((GSPL_IS_MAPPED(roughness.gaussian)
                            || GSPL_IS_MAPPED(roughness.polished)),
                           << "Missing Gaussian roughness or polished surface "
                              "from Unified model");
            CELER_VALIDATE((GSPL_IS_MAPPED(interaction.dielectric_dielectric)
                            || GSPL_IS_MAPPED(interaction.dielectric_metal)),
                           << "Missing ReflectionForm data for surface '"
                           << surf.GetName() << "'");

            // Expected empty maps
            CELER_VALIDATE(!GSPL_IS_MAPPED(roughness.smear),
                           << "Smear roughness is not used by the Unified "
                              "model and therefore should not be "
                              "assigned");
            break;

        default:
            CELER_LOG(error) << "G4OpticalSurfaceModel '" << to_cstring(model)
                             << "' not available";
            break;
    }

#undef GSPL_IS_MAPPED
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
