//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSurfacePhysicsLoader.cc
//---------------------------------------------------------------------------//
#include "GeantSurfacePhysicsLoader.hh"

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
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with Geant4 geometry data.
 */
GeantSurfacePhysicsLoader::GeantSurfacePhysicsLoader()
    : geo_(celeritas::geant_geo().lock())
{
    CELER_VALIDATE(geo_, << "global Geant4 geometry is not loaded");
}

//---------------------------------------------------------------------------//
/*!
 * Populate surface physics data.
 */
inp::SurfacePhysics GeantSurfacePhysicsLoader::operator()()
{
    inp::SurfacePhysics result;

    for (auto sid : range(SurfaceId(geo_->num_surfaces())))
    {
        auto const* g4log_surf = geo_->id_to_geant(sid);
        CELER_ASSERT(g4log_surf);
        auto* g4surf_prop = g4log_surf->GetSurfaceProperty();
        CELER_ASSERT(g4surf_prop);
        auto* g4opt_surf = dynamic_cast<G4OpticalSurface*>(g4surf_prop);
        CELER_ASSERT(g4opt_surf);
        auto const* g4mpt = g4opt_surf->GetMaterialPropertiesTable();
        CELER_ASSERT(g4mpt);
        detail::GeantMaterialPropertyGetter get_property{*g4mpt};

        try
        {
            this->insert_reflectivity(sid, *g4opt_surf, get_property, result);
            this->insert_roughness(sid, *g4opt_surf, result);
            this->insert_interaction(sid, get_property, *g4opt_surf, result);

            // Ensure that data is not incompatible with selected model
            this->validate_model(sid, *g4opt_surf, result);
        }
        catch (RuntimeError const& e)
        {
            CELER_VALIDATE(
                false,
                << "failed to convert surface " << g4opt_surf->GetName()
                << " with model " << to_cstring(g4opt_surf->GetModel())
                << " and finish " << to_cstring(g4opt_surf->GetFinish())
                << ": " << e.details().which << ", " << e.details().what);
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Collect reflectivity information from a given optical surface.
 */
void GeantSurfacePhysicsLoader::insert_reflectivity(
    SurfaceId sid,
    G4OpticalSurface const& surf,
    detail::GeantMaterialPropertyGetter& get_property,
    inp::SurfacePhysics& result)
{
    inp::ReflectivityModels refl_mods;
    if (!this->analytic_reflection_only(surf))
    {
        // Insert any model that includes user-defined grid reflectivity
        inp::ReflectionGrid refl_grid;
        get_property(&refl_grid.grid,
                     "REFLECTIVITY",
                     {ImportUnits::mev, ImportUnits::unitless});
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
void GeantSurfacePhysicsLoader::insert_roughness(SurfaceId sid,
                                                 G4OpticalSurface const& surf,
                                                 inp::SurfacePhysics& result)
{
    using G4OSM = G4OpticalSurfaceModel;

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
    SurfaceId sid,
    detail::GeantMaterialPropertyGetter& get_property,
    G4OpticalSurface const& surf,
    inp::SurfacePhysics& result)
{
    using G4ST = G4SurfaceType;

    inp::ReflectionForm refl_form;
    get_property(&refl_form.specular_lobe,
                 "SPECULARLOBECONSTANT",
                 {ImportUnits::mev, ImportUnits::unitless});
    get_property(&refl_form.specular_spike,
                 "SPECULARSPIKECONSTANT",
                 {ImportUnits::mev, ImportUnits::unitless});
    get_property(&refl_form.backscatter,
                 "BACKSCATTERCONSTANT",
                 {ImportUnits::mev, ImportUnits::unitless});
    refl_form.diffuse_lobe = this->calc_diffuse_lobe(refl_form);

    if (refl_form)
    {
        // ReflectionForm terms are correctly assigned; Add to interface type
        auto const interface_type = surf.GetType();
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
                CELER_LOG(error)
                    << "G4SurfaceType '" << to_cstring(interface_type)
                    << "' not available";
                break;
        }
    }
    else
    {
        CELER_LOG(error) << "inp::ReflectionForm incorrectly set up. Verify "
                            "that all parameters have the same grid sizes and "
                            "that their probability sums (for each energy "
                            "point in the grid) are equal to 1";
    }
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
 * Calculate diffuse lobe from the rest of the imported data.
 *
 * Since the total probability for all 4 properties is equal to one, the
 * diffuse lobe can be calculated by subtracting the other three.
 */
inp::Grid GeantSurfacePhysicsLoader::calc_diffuse_lobe(
    inp::ReflectionForm const& refl_form)
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
/*!
 * Ensure that a mapped optical surface does not hot have inconsistent model
 * data assigned to it.
 *
 * \note
 * - GLISUR
 *   - Roughness: uses polished or smear; Gaussian is never used.
 * - Unified
 *   - Roughness: uses Gaussian or polished; smear is never used.
 *   - ReflectiomForm: \c specular_spike , \c specular_lobe , \c backscatter .
 */
void GeantSurfacePhysicsLoader::validate_model(SurfaceId sid,
                                               G4OpticalSurface const& surf,
                                               inp::SurfacePhysics const& result)
{
    CELER_EXPECT(sid);
    CELER_EXPECT(result);
    using G4OSM = G4OpticalSurfaceModel;

#define GSPL_IS_MAPPED(MEMBER) (result.MEMBER.find(sid) != result.MEMBER.end())

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
