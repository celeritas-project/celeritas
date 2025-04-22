//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/MaterialParamsOutput.cc
//---------------------------------------------------------------------------//
#include "MaterialParamsOutput.hh"

#include <utility>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/LabelIO.json.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Types.hh"

#include "MaterialParams.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared material data.
 */
MaterialParamsOutput::MaterialParamsOutput(SPConstMaterialParams material)
    : material_(std::move(material))
{
    CELER_EXPECT(material_);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void MaterialParamsOutput::output(JsonPimpl* j) const
{
    using json = nlohmann::json;

    auto obj = json::object();
    auto units = json::object();

    // Unfold isotopes
    {
        auto label = json::array();
        auto atomic_number = json::array();
        auto atomic_mass_number = json::array();
        auto binding_energy = json::array();
        auto proton_loss_energy = json::array();
        auto neutron_loss_energy = json::array();
        auto nuclear_mass = json::array();

        for (auto id : range(IsotopeId{material_->num_isotopes()}))
        {
            IsotopeView const iso_view = material_->get(id);
            label.push_back(material_->id_to_label(id));
            atomic_number.push_back(iso_view.atomic_number().unchecked_get());
            atomic_mass_number.push_back(
                iso_view.atomic_mass_number().unchecked_get());
            binding_energy.push_back(iso_view.binding_energy().value());
            proton_loss_energy.push_back(iso_view.proton_loss_energy().value());
            neutron_loss_energy.push_back(
                iso_view.neutron_loss_energy().value());
            nuclear_mass.push_back(iso_view.nuclear_mass().value());
        }
        obj["isotopes"] = {
            {"label", std::move(label)},
            {"atomic_number", std::move(atomic_number)},
            {"atomic_mass_number", std::move(atomic_mass_number)},
            {"binding_energy", std::move(binding_energy)},
            {"proton_loss_energy", std::move(proton_loss_energy)},
            {"neutron_loss_energy", std::move(neutron_loss_energy)},
            {"nuclear_mass", std::move(nuclear_mass)},
        };
        units["binding_energy"]
            = accessor_unit_label<decltype(&IsotopeView::binding_energy)>();
        units["proton_loss_energy"]
            = accessor_unit_label<decltype(&IsotopeView::proton_loss_energy)>();
        units["neutron_loss_energy"]
            = accessor_unit_label<decltype(&IsotopeView::neutron_loss_energy)>();
        units["nuclear_mass"]
            = accessor_unit_label<decltype(&IsotopeView::nuclear_mass)>();
    }

    // Unfold elements
    {
        auto label = json::array();
        auto atomic_number = json::array();
        auto atomic_mass = json::array();
        auto coulomb_correction = json::array();
        auto mass_radiation_coeff = json::array();
        auto isotope_ids = json::array();
        auto isotope_fracs = json::array();

        for (auto id : range(ElementId{material_->num_elements()}))
        {
            ElementView const el_view = material_->get(id);
            label.push_back(material_->id_to_label(id));
            atomic_number.push_back(el_view.atomic_number().unchecked_get());
            atomic_mass.push_back(el_view.atomic_mass().value());
            coulomb_correction.push_back(el_view.coulomb_correction());
            mass_radiation_coeff.push_back(el_view.mass_radiation_coeff());

            // Save isotope ids and fractions
            auto el_isot_ids = json::array();
            auto el_isot_fracs = json::array();
            for (auto iso_comp : el_view.isotopes())
            {
                el_isot_ids.push_back(iso_comp.isotope.unchecked_get());
                el_isot_fracs.push_back(iso_comp.fraction);
            }
            isotope_ids.push_back(std::move(el_isot_ids));
            isotope_fracs.push_back(std::move(el_isot_fracs));
        }
        obj["elements"] = {
            {"label", std::move(label)},
            {"atomic_number", std::move(atomic_number)},
            {"atomic_mass", std::move(atomic_mass)},
            {"isotope_ids", std::move(isotope_ids)},
            {"isotope_fractions", std::move(isotope_fracs)},
            {"coulomb_correction", std::move(coulomb_correction)},
            {"mass_radiation_coeff", std::move(mass_radiation_coeff)},
        };
        units["atomic_mass"]
            = accessor_unit_label<decltype(&ElementView::atomic_mass)>();
    }

    // Unfold materials
    {
        auto label = json::array();
        auto number_density = json::array();
        auto temperature = json::array();
        auto matter_state = json::array();

        auto zeff = json::array();
        auto density = json::array();
        auto electron_density = json::array();
        auto radiation_length = json::array();
        auto mean_excitation_energy = json::array();

        auto element_id = json::array();
        auto element_frac = json::array();

        for (auto id : range(PhysMatId{material_->num_materials()}))
        {
            MaterialView const mat_view = material_->get(id);
            label.push_back(material_->id_to_label(id));
            number_density.push_back(mat_view.number_density());
            temperature.push_back(mat_view.temperature());
            matter_state.push_back(to_cstring(mat_view.matter_state()));

            // Save derivative data
            zeff.push_back(mat_view.zeff());
            density.push_back(mat_view.density());
            electron_density.push_back(mat_view.electron_density());
            radiation_length.push_back(mat_view.radiation_length());
            mean_excitation_energy.push_back(
                mat_view.mean_excitation_energy().value());

            // Save element ids and fractions
            auto elids = json::array();
            auto elfrac = json::array();
            for (MatElementComponent const& el : mat_view.elements())
            {
                elids.push_back(el.element.unchecked_get());
                elfrac.push_back(el.fraction);
            }
            element_id.push_back(std::move(elids));
            element_frac.push_back(std::move(elfrac));
        }
        obj["materials"] = {
            {"label", std::move(label)},
            {"number_density", std::move(number_density)},
            {"temperature", std::move(temperature)},
            {"matter_state", std::move(matter_state)},
            {"element_id", std::move(element_id)},
            {"element_frac", std::move(element_frac)},
            {"zeff", std::move(zeff)},
            {"density", std::move(density)},
            {"electron_density", std::move(electron_density)},
            {"radiation_length", std::move(radiation_length)},
            {"mean_excitation_energy", std::move(mean_excitation_energy)},
        };
        {
            units["mean_excitation_energy"] = accessor_unit_label<
                decltype(&MaterialView::mean_excitation_energy)>();
        }
    }

    obj["_units"] = std::move(units);
    j->obj = std::move(obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
