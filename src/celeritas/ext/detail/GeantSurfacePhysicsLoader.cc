//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantSurfacePhysicsLoader.cc
//---------------------------------------------------------------------------//
#include "GeantSurfacePhysicsLoader.hh"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <G4LogicalSurface.hh>
#include <G4OpticalSurface.hh>
#include <G4Version.hh>

#include "corecel/io/Logger.hh"
#include "geocel/SurfaceParams.hh"

namespace celeritas
{
namespace detail
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

    static std::unordered_map<G4SurfaceType, const char*> const names = {
        GSPL_ST_PAIR(dielectric_metal),
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
    if (auto iter = names.find(value); iter != names.end())
    {
        return names.find(value)->second;
    }
    return "UNKNOWN";

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

    static std::unordered_map<G4OpticalSurfaceModel, const char*> const names
        = {GSPL_OSM_PAIR(glisur),
           GSPL_OSM_PAIR(unified),
           GSPL_OSM_PAIR(LUT),
           GSPL_OSM_PAIR(DAVIS),
           GSPL_OSM_PAIR(dichroic)};

    if (auto iter = names.find(value); iter != names.end())
    {
        return names.find(value)->second;
    }
    return "UNKNOWN";

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

    static std::unordered_map<G4OpticalSurfaceFinish, const char*> const names
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

    if (auto iter = names.find(value); iter != names.end())
    {
        return names.find(value)->second;
    }
    return "UNKNOWN";

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
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with \c SurfacePhysics input to be filled by \c operator() .
 */
GeantSurfacePhysicsLoader::GeantSurfacePhysicsLoader(inp::SurfacePhysics& result)
    : result_(result)
{
}

//---------------------------------------------------------------------------//
/*!
 * Populate surface physics data for a given \c SurfaceId .
 */
void GeantSurfacePhysicsLoader::operator()(SurfaceId sid)
{
    CELER_EXPECT(sid);
    detail::GeantSurfacePhysicsHelper helper(sid);
    try
    {
        this->insert_reflectivity(helper);
        this->insert_roughness(helper);
        this->insert_interaction(helper);
        this->validate_model(helper);  // Verify model requirements
    }
    catch (RuntimeError const& e)
    {
        throw;
    }

    CELER_LOG(debug) << "Inserted surface id " << sid.unchecked_get()
                     << " with " << to_cstring(helper.surface().GetModel())
                     << " model and "
                     << to_cstring(helper.surface().GetFinish()) << " finish";
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Collect reflectivity information from a given optical surface.
 */
void GeantSurfacePhysicsLoader::insert_reflectivity(
    detail::GeantSurfacePhysicsHelper& helper)
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
    result_.reflectivity = std::move(refl_mods);
    CELER_ENSURE(result_.reflectivity);
}

//---------------------------------------------------------------------------//
/*!
 * Collect roughness information from a given optical surface.
 */
void GeantSurfacePhysicsLoader::insert_roughness(
    detail::GeantSurfacePhysicsHelper& helper)
{
    using G4OSM = G4OpticalSurfaceModel;
    using G4OSF = G4OpticalSurfaceFinish;

    auto sid = helper.surface_id();
    auto const& surf = helper.surface();
    auto const g4model = surf.GetModel();
    switch (g4model)
    {
        case G4OSM::glisur: {
            if (surf.GetFinish() == G4OSF::polished)
            {
                // Perfectly polished surface
                result_.roughness.polished.insert({sid, inp::Polished{}});
            }
            else
            {
                // Smearing is available (surf.GetFinish() == G4OSF::ground)
                // Celeritas' roughness is the complement of Geant4 polish
                inp::SmearRoughness smear{real_type{1} - surf.GetPolish()};
                CELER_VALIDATE(
                    smear, << "Smear roughness must be within [0, 1] range");
                result_.roughness.smear.insert({sid, std::move(smear)});
            }
            break;
        }

        case G4OSM::unified: {
            // Insert Gaussian if available
            inp::GaussianRoughness gauss;
            gauss.sigma_alpha = surf.GetSigmaAlpha();
            if (gauss)
            {
                result_.roughness.gaussian.insert({sid, std::move(gauss)});
            }
            break;
        }

        default:
            CELER_LOG(error) << "G4OpticalSurfaceModel '"
                             << to_cstring(g4model) << "' not available";
            break;
    }
    CELER_ENSURE(result_.roughness);
}

//---------------------------------------------------------------------------//
/*!
 * Collect interaction information from a given optical surface.
 */
void GeantSurfacePhysicsLoader::insert_interaction(
    detail::GeantSurfacePhysicsHelper& helper)
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
    CELER_ASSERT(refl_form);

    // Verify unity of reflection form parameters
    GSPL_VALIDATE_UNITY(refl_form.specular_spike);
    GSPL_VALIDATE_UNITY(refl_form.specular_lobe);
    GSPL_VALIDATE_UNITY(refl_form.backscatter);

    // ReflectionForm terms are correctly assigned; add to interface type
    auto sid = helper.surface_id();
    auto const interface_type = helper.surface().GetType();
    switch (interface_type)
    {
        case G4ST::dielectric_dielectric:
            result_.interaction.dielectric_dielectric.insert(
                {sid, std::move(refl_form)});
            break;
        case G4ST::dielectric_metal:
            result_.interaction.dielectric_metal.insert(
                {sid, std::move(refl_form)});
            break;
        default:
            CELER_LOG(error) << "G4SurfaceType '" << to_cstring(interface_type)
                             << "' not available";
            break;
    }

#undef GSPL_VALIDATE_UNITY

    CELER_ENSURE(result_.interaction);
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
    detail::GeantSurfacePhysicsHelper& helper) const
{
    using G4OSM = G4OpticalSurfaceModel;

#define GSPL_IS_MAPPED(MEMBER) \
    (result_.MEMBER.find(sid) != result_.MEMBER.end())

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
                              "for the Unified model");
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
}  // namespace detail
}  // namespace celeritas
