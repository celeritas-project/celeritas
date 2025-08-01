//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantSurfacePhysicsLoader.cc
//---------------------------------------------------------------------------//
#include "GeantSurfacePhysicsLoader.hh"

#include <G4LogicalSurface.hh>
#include <G4OpticalSurface.hh>
#include <G4Version.hh>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"

#include "GeantSurfacePhysicsHelper.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
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
    return std::all_of(grid.y.begin(), grid.y.end(), [](real_type const& val) {
        return val >= 0 && val <= 1;
    });
}

//---------------------------------------------------------------------------//
/*!
 * Populate all \c ReflectionForm parameters for the Unified model.
 */
inp::ReflectionForm load_unified_refl_form(GeantSurfacePhysicsHelper& helper)
{
    inp::ReflectionForm refl_form;
    helper.get_property(&refl_form.specular_lobe, "SPECULARLOBECONSTANT");
    helper.get_property(&refl_form.specular_spike, "SPECULARSPIKECONSTANT");
    helper.get_property(&refl_form.backscatter, "BACKSCATTERCONSTANT");
    CELER_ASSERT(refl_form);

// Verify unity of reflection form parameters
#define GSPL_VALIDATE_UNITY(PARAM)                           \
    CELER_VALIDATE(unity(PARAM),                             \
                   << "ReflectionForm parameter '" << #PARAM \
                   << "' is not within [0, 1] range")
    GSPL_VALIDATE_UNITY(refl_form.specular_spike);
    GSPL_VALIDATE_UNITY(refl_form.specular_lobe);
    GSPL_VALIDATE_UNITY(refl_form.backscatter);
#undef GSPL_VALIDATE_UNITY

    return refl_form;
}

//---------------------------------------------------------------------------//
/*!
 * Populate a \c inp::ReflectionGrid object for a given surface
 */
inp::ReflectionGrid load_refl_grid(GeantSurfacePhysicsHelper& helper)
{
    inp::ReflectionGrid result;
    helper.get_property(&result.grid, "REFLECTIVITY");
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Insert grid and analytic reflectivity modes into a \c inp::SurfacePhysics
 * object.
 */
void insert_grid_analytic_reflectivities(inp::SurfacePhysics& inp,
                                         GeantSurfacePhysicsHelper& helper)
{
    auto const sid = helper.surface_id();
    inp.reflectivity.analytic.insert({sid, inp::ReflectionAnalytic{}});
    inp.reflectivity.grid.insert({sid, load_refl_grid(helper)});
}

//---------------------------------------------------------------------------//
/*!
 * Throw error message based on optical surface physics selection.
 */
std::string throw_error_msg(G4OpticalSurface const& surf)
{
    return "Surface " + surf.GetName() + " with surface finish '"
           + to_cstring(surf.GetFinish()) + "' is not compatible with '"
           + to_cstring(surf.GetType()) + "' surface type on the "
           + to_cstring(surf.GetModel()) + " model";
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
    using G4OSM = G4OpticalSurfaceModel;

    GeantSurfacePhysicsHelper helper(sid);
    auto const& surf = helper.surface();
    auto const model = surf.GetModel();
    switch (model)
    {
        case G4OSM::glisur:
            this->insert_glisur(helper);
            break;
        case G4OSM::unified:
            this->insert_unified(helper);
            break;
        default:
            CELER_NOT_IMPLEMENTED("Model " + std::string(to_cstring(model)));
    }
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Insert GLISUR model surface.
 */
void GeantSurfacePhysicsLoader::insert_glisur(GeantSurfacePhysicsHelper& helper)
{
    using G4ST = G4SurfaceType;
    using G4OSF = G4OpticalSurfaceFinish;

    auto const& surf = helper.surface();
    auto sid = helper.surface_id();
    auto const type = surf.GetType();
    auto const finish = surf.GetFinish();
    switch (finish)
    {
        case G4OSF::polished: {
            // Insert polished surface with specular spike reflection mode
            result_.roughness.polished.insert({sid, inp::NoRoughness{}});

            std::pair<SurfaceId, inp::ReflectionForm> pair{
                sid, inp::ReflectionForm::from_spike()};

            (type == G4ST::dielectric_dielectric)
                ? result_.interaction.dielectric_dielectric.insert(pair)
                : result_.interaction.dielectric_metal.insert(pair);
            break;
        }

        case G4OSF::ground: {
            // Insert smear surface with specular lobe reflection mode
            inp::ReflectionForm refl_form;
            real_type roughness = real_type{1} - surf.GetPolish();
            result_.roughness.smear.insert(
                {sid, inp::SmearRoughness{roughness}});

            std::pair<SurfaceId, inp::ReflectionForm> pair{
                sid, inp::ReflectionForm::from_lobe()};

            (type == G4ST::dielectric_dielectric)
                ? result_.interaction.dielectric_dielectric.insert(pair)
                : result_.interaction.dielectric_metal.insert(pair);
            break;
        }

        default:
            CELER_VALIDATE(false, << throw_error_msg(surf));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Insert unified model surface.
 *
 * Data is populated according to the table from Celeritas issue #1512:
 * https://github.com/celeritas-project/celeritas/issues/1512#issuecomment-3019564068
 */
void GeantSurfacePhysicsLoader::insert_unified(GeantSurfacePhysicsHelper& helper)
{
    using G4ST = G4SurfaceType;
    using G4OSF = G4OpticalSurfaceFinish;

    auto const& surf = helper.surface();
    auto sid = helper.surface_id();
    auto const type = surf.GetType();
    auto const finish = surf.GetFinish();
    switch (finish)
    {
        //// Used by dielectric-dielectric and dielectric-metal interfaces ////
        case G4OSF::polished: {
            result_.roughness.polished.insert({sid, inp::NoRoughness{}});
            insert_grid_analytic_reflectivities(result_, helper);

            // Insert interaction based on surface type
            (type == G4ST::dielectric_dielectric)
                ? result_.interaction.dielectric_dielectric.insert(
                      {sid, inp::ReflectionForm::from_spike()})
                : result_.interaction.dielectric_metal.insert(
                      {sid, load_unified_refl_form(helper)});

            break;
        }

        case G4OSF::ground: {
            result_.roughness.gaussian.insert(
                {sid, inp::GaussianRoughness{surf.GetSigmaAlpha()}});
            insert_grid_analytic_reflectivities(result_, helper);

            // Insert interaction based on surface type
            (type == G4ST::dielectric_dielectric)
                ? result_.interaction.dielectric_dielectric.insert(
                      {sid, inp::ReflectionForm::from_spike()})
                : result_.interaction.dielectric_metal.insert(
                      {sid, load_unified_refl_form(helper)});

            break;
        }

        //// Only available to dielectric-dielectric interfaces ////
        case G4OSF::polishedfrontpainted: {
            result_.roughness.polished.insert({sid, inp::NoRoughness{}});
            insert_grid_analytic_reflectivities(result_, helper);

            // Insert specular spike reflection form
            result_.interaction.dielectric_dielectric.insert(
                {sid, inp::ReflectionForm::from_spike()});
            break;
        }

        case G4OSF::groundfrontpainted: {
            result_.roughness.gaussian.insert(
                {sid, inp::GaussianRoughness{surf.GetSigmaAlpha()}});
            insert_grid_analytic_reflectivities(result_, helper);

            // Insert Lambertian reflection form
            result_.interaction.dielectric_dielectric.insert(
                {sid, inp::ReflectionForm::from_lambertian()});
            break;
        }

        case G4OSF::polishedbackpainted: {
            // Equivalent to layer 0
            result_.roughness.gaussian.insert(
                {sid, inp::GaussianRoughness{surf.GetSigmaAlpha()}});
            // Equivalent to layer 1
            result_.roughness.polished.insert({sid, inp::NoRoughness{}});
            // Analytic for layer 0; grid for layer 1
            insert_grid_analytic_reflectivities(result_, helper);
            // Insert interface
            // Layer 0 uses any reflection form; Layer 1 uses specular spike
            result_.interaction.dielectric_dielectric.insert(
                {sid, load_unified_refl_form(helper)});
            break;
        }

        case G4OSF::groundbackpainted: {
            // Equivalent to layer 0: Gaussian, analytic reflection
            result_.roughness.gaussian.insert(
                {sid, inp::GaussianRoughness{surf.GetSigmaAlpha()}});
            // Equivalent to layer 1: Polished, grid, Lambertian reflection
            result_.roughness.polished.insert({sid, inp::NoRoughness{}});
            // Analytic for layer 0; grid for layer 1
            insert_grid_analytic_reflectivities(result_, helper);
            // Insert interface
            // Layer 0 uses all reflections; Layer 1 uses Lambertian
            result_.interaction.dielectric_dielectric.insert(
                {sid, load_unified_refl_form(helper)});
            break;
        }
        default:
            CELER_VALIDATE(false, << throw_error_msg(surf));
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
