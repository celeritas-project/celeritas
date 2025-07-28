//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSurfacePhysicsLoader.cc
//---------------------------------------------------------------------------//
#include "GeantSurfacePhysicsLoader.hh"

#include <G4LogicalSurface.hh>
#include <G4OpticalSurface.hh>
#include <G4Version.hh>

#include "corecel/io/Logger.hh"
#include "geocel/SurfaceParams.hh"

namespace celeritas
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
    static std::map<G4SurfaceType, std::string> const to_cstring_impl = {
        {G4SurfaceType::dielectric_metal, "dielectric_metal"},
        {G4SurfaceType::dielectric_dielectric, "dielectric_dielectric"},
        {G4SurfaceType::dielectric_LUT, "dielectric_LUT"},
        {G4SurfaceType::dielectric_LUTDAVIS, "dielectric_LUTDAVIS"},
        {G4SurfaceType::dielectric_dichroic, "dielectric_dichroic"},
        {G4SurfaceType::firsov, "firsov"},
        {G4SurfaceType::x_ray, "x_ray"},
#if G4VERSION_NUMBER >= 1110
        {G4SurfaceType::coated, "coated"}
#endif
    };
    return to_cstring_impl.find(value)->second.c_str();
}
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the \c G4OpticalSurfaceModel selection.
 */
char const* to_cstring(G4OpticalSurfaceModel value)
{
    static std::map<G4OpticalSurfaceModel, std::string> const to_cstring_impl
        = {{G4OpticalSurfaceModel::glisur, "glisur"},
           {G4OpticalSurfaceModel::unified, "unified"},
           {G4OpticalSurfaceModel::LUT, "LUT"},
           {G4OpticalSurfaceModel::DAVIS, "DAVIS"},
           {G4OpticalSurfaceModel::dichroic, "dichroic"}};
    return to_cstring_impl.find(value)->second.c_str();
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the \c G4OpticalSurfaceFinish selection.
 */
char const* to_cstring(G4OpticalSurfaceFinish value)
{
    static std::map<G4OpticalSurfaceFinish, std::string> const to_cstring_impl = {
        {G4OpticalSurfaceFinish::polished, "polished"},
        {G4OpticalSurfaceFinish::polishedfrontpainted, "polishedfrontpainted"},
        {G4OpticalSurfaceFinish::polishedbackpainted, "polishedbackpainted"},

        {G4OpticalSurfaceFinish::ground, "ground"},
        {G4OpticalSurfaceFinish::groundfrontpainted, "groundfrontpainted"},
        {G4OpticalSurfaceFinish::groundbackpainted, "groundbackpainted"},

        {G4OpticalSurfaceFinish::polishedlumirrorair, "polishedlumirrorair"},
        {G4OpticalSurfaceFinish::polishedlumirrorglue, "polishedlumirrorglue"},
        {G4OpticalSurfaceFinish::polishedair, "polishedair"},
        {G4OpticalSurfaceFinish::polishedteflonair, "polishedteflonair"},
        {G4OpticalSurfaceFinish::polishedtioair, "polishedtioair"},
        {G4OpticalSurfaceFinish::polishedtyvekair, "polishedtyvekair"},
        {G4OpticalSurfaceFinish::polishedvm2000air, "polishedvm2000air"},
        {G4OpticalSurfaceFinish::polishedvm2000glue, "polishedvm2000glue"},

        {G4OpticalSurfaceFinish::etchedlumirrorair, "etchedlumirrorair"},
        {G4OpticalSurfaceFinish::etchedlumirrorglue, "etchedlumirrorglue"},
        {G4OpticalSurfaceFinish::etchedair, "etchedair"},
        {G4OpticalSurfaceFinish::etchedteflonair, "etchedteflonair"},
        {G4OpticalSurfaceFinish::etchedtioair, "etchedtioair"},
        {G4OpticalSurfaceFinish::etchedtyvekair, "etchedtyvekair"},
        {G4OpticalSurfaceFinish::etchedvm2000air, "etchedvm2000air"},
        {G4OpticalSurfaceFinish::etchedvm2000glue, "etchedvm2000glue"},

        {G4OpticalSurfaceFinish::groundlumirrorair, "groundlumirrorair"},
        {G4OpticalSurfaceFinish::groundlumirrorglue, "groundlumirrorglue"},
        {G4OpticalSurfaceFinish::groundair, "groundair"},
        {G4OpticalSurfaceFinish::groundteflonair, "groundteflonair"},
        {G4OpticalSurfaceFinish::groundtioair, "groundtioair"},
        {G4OpticalSurfaceFinish::groundtyvekair, "groundtyvekair"},
        {G4OpticalSurfaceFinish::groundvm2000air, "groundvm2000air"},
        {G4OpticalSurfaceFinish::groundvm2000glue, "groundvm2000glue"},

        {G4OpticalSurfaceFinish::Rough_LUT, "Rough_LUT"},
        {G4OpticalSurfaceFinish::RoughTeflon_LUT, "RoughTeflon_LUT"},
        {G4OpticalSurfaceFinish::RoughESR_LUT, "RoughESR_LUT"},
        {G4OpticalSurfaceFinish::RoughESRGrease_LUT, "RoughESRGrease_LUT"},

        {G4OpticalSurfaceFinish::Polished_LUT, "Polished_LUT"},
        {G4OpticalSurfaceFinish::PolishedTeflon_LUT, "PolishedTeflon_LUT"},
        {G4OpticalSurfaceFinish::PolishedESR_LUT, "PolishedESR_LUT"},
        {G4OpticalSurfaceFinish::PolishedESRGrease_LUT,
         "PolishedESRGrease_LUT"},

        {G4OpticalSurfaceFinish::Detector_LUT, "Detector_LUT"},
    };
    return to_cstring_impl.find(value)->second.c_str();
}

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
        auto const* g4mpt = g4opt_surf->GetMaterialPropertiesTable();
        CELER_ASSERT(g4mpt);
        detail::GeantMaterialPropertyGetter get_property{*g4mpt};

        try
        {
            result.names.insert({sid, g4opt_surf->GetName()});
            this->insert_reflectivity(sid, *g4opt_surf, get_property, result);
            this->insert_roughness(sid, *g4opt_surf, result);
            this->insert_interaction(sid, get_property, *g4opt_surf, result);
            this->insert_efficiency(sid, get_property, result);

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

    if (g4model == G4OSM::glisur)
    {
        // Get GLISUR surface polish
        real_type polishness = surf.GetPolish();
        if (polishness == 1)
        {
            // Perfectly polished surface
            result.roughness.polished.insert({sid, inp::Polished{}});
        }
        else
        {
            // Smearing is available
            inp::SmearRoughness smear{polishness};
            CELER_ASSERT(smear);
            result.roughness.smear.insert({sid, std::move(smear)});
        }
    }

    else if (g4model == G4OSM::unified)
    {
        // Insert Gaussian if available
        inp::GaussianRoughness gauss;
        gauss.sigma_alpha = surf.GetSigmaAlpha();
        if (gauss)
        {
            result.roughness.gaussian.insert({sid, std::move(gauss)});
        }
    }

    else
    {
        CELER_LOG(error) << "G4OpticalSurfaceModel '" << to_cstring(g4model)
                         << "' not available";
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
    get_property(&refl_form.back_scatter,
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
 * Collect detection efficiency from a given optical surface.
 */
void GeantSurfacePhysicsLoader::insert_efficiency(
    SurfaceId sid,
    detail::GeantMaterialPropertyGetter& get_property,
    inp::SurfacePhysics& result)
{
    inp::Grid eff;
    get_property(&eff, "EFFICIENCY", {ImportUnits::mev, ImportUnits::unitless});
    result.efficiency.insert({sid, eff});
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
    auto const& bc = refl_form.back_scatter;
    auto const size = sl.x.size();
    CELER_ASSERT(ss.x.size() == size && bc.x.size() == size);

    inp::Grid result;
    result.x = sl.x;
    result.y.resize(size);
    for (auto i : range(size))
    {
        // diffuse_lobe = 1 - specular_lobe - specular_spike - back_scatter
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
 *   - ReflectiomForm: \c specular_spike , \c specular_lobe , \c back_scatter .
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
                           << result.names.find(sid)->second << "'");

            // Expected empty maps
            CELER_VALIDATE(!GSPL_IS_MAPPED(roughness.smear),
                           << "Smear roughness is not used by the Unified "
                              "model and therefore should not be assigned");
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
