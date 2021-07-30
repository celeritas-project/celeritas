//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialParams.cc
//---------------------------------------------------------------------------//
#include "MaterialParams.hh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include "detail/Utils.hh"
#include "base/Algorithms.hh"
#include "base/CollectionBuilder.hh"
#include "base/Range.hh"
#include "base/SoftEqual.hh"
#include "base/SpanRemapper.hh"
#include "comm/Logger.hh"
#include "io/ImportData.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Safely switch between MatterState [MaterialParams.hh] and
 * ImportMaterialState [ImportMaterial.hh].
 */
MatterState to_matter_state(const ImportMaterialState state)
{
    switch (state)
    {
        case ImportMaterialState::not_defined:
            return MatterState::unspecified;
        case ImportMaterialState::solid:
            return MatterState::solid;
        case ImportMaterialState::liquid:
            return MatterState::liquid;
        case ImportMaterialState::gas:
            return MatterState::gas;
    }
    CELER_ASSERT_UNREACHABLE();
}
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<MaterialParams>
MaterialParams::from_import(const ImportData& data)
{
    CELER_EXPECT(data);

    // Create MaterialParams input for its constructor
    MaterialParams::Input input;

    // Populate input.elements
    for (const auto& element : data.elements)
    {
        MaterialParams::ElementInput element_params;
        element_params.atomic_number = element.atomic_number;
        element_params.atomic_mass   = units::AmuMass{element.atomic_mass};
        element_params.name          = element.name;

        input.elements.push_back(element_params);
    }

    // Populate input.materials
    for (const auto& material : data.materials)
    {
        MaterialParams::MaterialInput material_params;
        material_params.name           = material.name;
        material_params.temperature    = material.temperature;
        material_params.number_density = material.number_density;
        material_params.matter_state   = to_matter_state(material.state);

        for (const auto& elem_comp : material.elements)
        {
            ElementId elem_def_id{elem_comp.element_id};

            // Populate MaterialParams number fractions
            material_params.elements_fractions.push_back(
                {elem_def_id, elem_comp.number_fraction});
        }
        input.materials.push_back(material_params);
    }

    // Construct MaterialParams and return it as a shared_ptr
    MaterialParams materials(input);
    return std::make_shared<MaterialParams>(std::move(materials));
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a vector of material definitions.
 */
MaterialParams::MaterialParams(const Input& inp)
{
    CELER_EXPECT(!inp.materials.empty());

    // Build elements and materials on host.
    HostValue host_data;
    for (const auto& el : inp.elements)
    {
        this->append_element_def(el, &host_data);
    }
    for (const auto& mat : inp.materials)
    {
        this->append_material_def(mat, &host_data);
    }

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<MaterialParamsData>{std::move(host_data)};

    CELER_ENSURE(this->data_);
    CELER_ENSURE(this->host_pointers().elements.size() == inp.elements.size());
    CELER_ENSURE(this->host_pointers().materials.size()
                 == inp.materials.size());
    CELER_ENSURE(elnames_.size() == inp.elements.size());
    CELER_ENSURE(matnames_.size() == inp.materials.size());
}

//---------------------------------------------------------------------------//
// IMPLEMENTATION
//---------------------------------------------------------------------------//
/*!
 * Convert an element input to an element definition and store.
 *
 * This adds computed quantities in addition to the input values. The result
 * is pushed back onto the host list of stored elements.
 */
void MaterialParams::append_element_def(const ElementInput& inp,
                                        HostValue*          host_data)
{
    CELER_EXPECT(inp.atomic_number > 0);
    CELER_EXPECT(inp.atomic_mass > zero_quantity());

    ElementDef result;

    // Copy basic properties
    result.atomic_number = inp.atomic_number;
    result.atomic_mass   = inp.atomic_mass;

    // Calculate various factors of the atomic number
    const real_type z_real = result.atomic_number;
    result.cbrt_z          = std::cbrt(z_real);
    result.cbrt_zzp        = std::cbrt(z_real * (z_real + 1));
    result.log_z           = std::log(z_real);
    result.coulomb_correction
        = detail::calc_coulomb_correction(result.atomic_number);
    result.mass_radiation_coeff = detail::calc_mass_rad_coeff(result);

    elnames_.push_back(inp.name);

    // Add to host vector
    make_builder(&host_data->elements).push_back(result);
}

//---------------------------------------------------------------------------//
/*!
 * Process and store element components to the internal list.
 *
 * \todo It's the caller's responsibility to ensure that element IDs
 * aren't duplicated.
 */
ItemRange<MatElementComponent>
MaterialParams::extend_elcomponents(const MaterialInput& inp,
                                    HostValue*           host_data) const
{
    CELER_EXPECT(host_data);
    // Allocate material components
    std::vector<MatElementComponent> components(inp.elements_fractions.size());

    // Store number fractions
    real_type norm = 0.0;
    for (auto i : range(inp.elements_fractions.size()))
    {
        CELER_EXPECT(inp.elements_fractions[i].first
                     < host_data->elements.size());
        CELER_EXPECT(inp.elements_fractions[i].second >= 0);
        // Store number fraction
        components[i].element  = inp.elements_fractions[i].first;
        components[i].fraction = inp.elements_fractions[i].second;
        // Add fractions to verify unity
        norm += inp.elements_fractions[i].second;
    }

    // Renormalize component fractions that are not unity and log them
    if (!inp.elements_fractions.empty() && !soft_equal(norm, 1.0))
    {
        CELER_LOG(warning) << "Element component fractions for `" << inp.name
                           << "` should sum to 1 but instead sum to " << norm
                           << " (difference = " << norm - 1 << ")";

        // Normalize
        norm                      = 1.0 / norm;
        real_type total_fractions = 0;
        for (MatElementComponent& comp : components)
        {
            comp.fraction *= norm;
            total_fractions += comp.fraction;
        }
        CELER_ASSERT(soft_equal(total_fractions, 1.0));
    }

    // Sort elements by increasing element ID for improved access
    std::sort(
        components.begin(),
        components.end(),
        [](const MatElementComponent& lhs, const MatElementComponent& rhs) {
            return lhs.element < rhs.element;
        });

    return make_builder(&host_data->elcomponents)
        .insert_back(components.begin(), components.end());
}

//---------------------------------------------------------------------------//
/*!
 * Convert an material input to an material definition and store.
 */
void MaterialParams::append_material_def(const MaterialInput& inp,
                                         HostValue*           host_data)
{
    CELER_EXPECT(inp.number_density >= 0);
    CELER_EXPECT((inp.number_density == 0) == inp.elements_fractions.empty());
    CELER_EXPECT(host_data);

    auto iter_inserted = matname_to_id_.insert(
        {inp.name, MaterialId(host_data->materials.size())});

    std::string mat_name = inp.name;

    if (!iter_inserted.second)
    {
        // Insertion failed due to duplicate material name
        // Create unique material name by concatenating its name and MaterialId
        mat_name = inp.name + "_" + std::to_string(host_data->materials.size());

        CELER_LOG(info)
            << "Material name " << inp.name << " already exists with id "
            << iter_inserted.second
            << ". Created new unique name identifier using its id: "
            << mat_name << ".";

        auto iter_reinserted = matname_to_id_.insert(
            {mat_name, MaterialId(host_data->materials.size())});

        CELER_ASSERT(iter_reinserted.second);
    }

    MaterialDef result;
    // Copy basic properties
    result.number_density = inp.number_density;
    result.temperature    = inp.temperature;
    result.matter_state   = inp.matter_state;
    result.elements       = this->extend_elcomponents(inp, host_data);

    /*!
     * Calculate derived quantities: density, electron density, and rad length
     *
     * NOTE: Electron density calculation may need to be updated for solids.
     */
    real_type avg_amu_mass        = 0;
    real_type avg_z               = 0;
    real_type rad_coeff           = 0;
    real_type log_mean_exc_energy = 0;
    for (const MatElementComponent& comp :
         host_data->elcomponents[result.elements])
    {
        CELER_ASSERT(comp.element < host_data->elements.size());
        const ElementDef& el = host_data->elements[comp.element];

        avg_amu_mass += comp.fraction * el.atomic_mass.value();
        avg_z += comp.fraction * el.atomic_number;
        rad_coeff += comp.fraction * el.mass_radiation_coeff;
        log_mean_exc_energy
            += comp.fraction * el.atomic_number
               * std::log(
                   detail::get_mean_excitation_energy(el.atomic_number).value());
    }
    result.density = result.number_density * avg_amu_mass
                     * constants::atomic_mass;
    result.electron_density = result.number_density * avg_z;
    result.rad_length       = 1 / (rad_coeff * result.density);
    log_mean_exc_energy     = avg_z > 0 ? log_mean_exc_energy / avg_z
                                    : -numeric_limits<real_type>::infinity();
    result.log_mean_exc_energy = units::LogMevEnergy{log_mean_exc_energy};
    result.mean_exc_energy = units::MevEnergy{std::exp(log_mean_exc_energy)};

    // Calculate the parameters for the energy loss fluctuation model (see
    // Geant3 PHYS332 section 2.4 and physics reference manual section 7.3.2)
    result.fluct.osc_strength[1]   = avg_z > 2 ? 2 / avg_z : 0;
    result.fluct.osc_strength[0]   = 1 - result.fluct.osc_strength[1];
    result.fluct.binding_energy[1] = 1e-5 * ipow<2>(avg_z);
    result.fluct.binding_energy[0]
        = std::pow(result.mean_exc_energy.value()
                       / std::pow(result.fluct.binding_energy[1],
                                  result.fluct.osc_strength[1]),
                   1 / result.fluct.osc_strength[0]);
    result.fluct.log_binding_energy[1]
        = std::log(result.fluct.binding_energy[1]);
    result.fluct.log_binding_energy[0]
        = std::log(result.fluct.binding_energy[0]);

    // Add to host vector
    make_builder(&host_data->materials).push_back(result);
    matnames_.push_back(mat_name);

    // Update maximum number of materials
    host_data->max_element_components
        = std::max(host_data->max_element_components, result.elements.size());

    CELER_ENSURE(result.number_density >= 0);
    CELER_ENSURE(result.temperature >= 0);
    CELER_ENSURE((result.density > 0) == (inp.number_density > 0));
    CELER_ENSURE((result.electron_density > 0) == (inp.number_density > 0));
    CELER_ENSURE(result.rad_length > 0);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
