//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/MaterialParams.cc
//---------------------------------------------------------------------------//
#include "MaterialParams.hh"

#include <algorithm>
#include <cmath>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportData.hh"

#include "MaterialData.hh"  // IWYU pragma: associated
#include "detail/Utils.hh"

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
MatterState to_matter_state(ImportMaterialState state)
{
    switch (state)
    {
        case ImportMaterialState::other:
            return MatterState::unspecified;
        case ImportMaterialState::solid:
            return MatterState::solid;
        case ImportMaterialState::liquid:
            return MatterState::liquid;
        case ImportMaterialState::gas:
            return MatterState::gas;
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<MaterialParams>
MaterialParams::from_import(ImportData const& data)
{
    CELER_EXPECT(!data.materials.empty());

    // Create MaterialParams input for its constructor
    MaterialParams::Input input;

    // Populate input.elements
    for (auto const& element : data.elements)
    {
        MaterialParams::ElementInput element_params;
        element_params.atomic_number = AtomicNumber{element.atomic_number};
        element_params.atomic_mass = units::AmuMass{element.atomic_mass};
        element_params.label = Label::from_geant(element.name);

        input.elements.push_back(element_params);
    }

    // Populate input.materials
    for (auto const& material : data.materials)
    {
        MaterialParams::MaterialInput material_params;
        material_params.temperature = material.temperature;
        material_params.number_density = material.number_density;
        material_params.matter_state = to_matter_state(material.state);
        material_params.label = Label::from_geant(material.name);

        for (auto const& elem_comp : material.elements)
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
MaterialParams::MaterialParams(Input const& inp)
{
    CELER_EXPECT(!inp.materials.empty());

    ScopedMem record_mem("MaterialParams.construct");

    // Build elements and materials on host.
    HostValue host_data;

    std::vector<Label> el_labels(inp.elements.size());
    for (auto i : range(inp.elements.size()))
    {
        el_labels[i] = inp.elements[i].label;
        this->append_element_def(inp.elements[i], &host_data);
    }
    el_labels_ = LabelIdMultiMap<ElementId>(std::move(el_labels));

    std::vector<Label> mat_labels(inp.materials.size());
    for (auto i : range(inp.materials.size()))
    {
        mat_labels[i] = inp.materials[i].label;
        this->append_material_def(inp.materials[i], &host_data);
    }
    mat_labels_ = LabelIdMultiMap<MaterialId>(std::move(mat_labels));

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<MaterialParamsData>{std::move(host_data)};

    CELER_ENSURE(this->data_);
    CELER_ENSURE(this->host_ref().elements.size() == inp.elements.size());
    CELER_ENSURE(this->host_ref().materials.size() == inp.materials.size());
    CELER_ENSURE(el_labels_.size() == inp.elements.size());
    CELER_ENSURE(mat_labels_.size() == inp.materials.size());
}

//---------------------------------------------------------------------------//
/*!
 * Get the label of a material.
 */
Label const& MaterialParams::id_to_label(MaterialId mat) const
{
    CELER_EXPECT(mat < mat_labels_.size());
    return mat_labels_.get(mat);
}

//---------------------------------------------------------------------------//
/*!
 * Locate the material ID corresponding to a label.
 *
 * If the label isn't among the materials, a null ID will be returned.
 */
MaterialId MaterialParams::find_material(std::string const& name) const
{
    auto result = mat_labels_.find_all(name);
    if (result.empty())
        return {};
    CELER_VALIDATE(result.size() == 1,
                   << "material '" << name << "' is not unique");
    return result.front();
}

//---------------------------------------------------------------------------//
/*!
 * Get zero or more material IDs corresponding to a name.
 *
 * This is useful for materials that are repeated with different
 * uniquifying 'extensions'.
 */
auto MaterialParams::find_materials(std::string const& name) const
    -> SpanConstMaterialId
{
    return mat_labels_.find_all(name);
}

//---------------------------------------------------------------------------//
/*!
 * Get the label of a element.
 */
Label const& MaterialParams::id_to_label(ElementId el) const
{
    CELER_EXPECT(el < el_labels_.size());
    return el_labels_.get(el);
}

//---------------------------------------------------------------------------//
/*!
 * Locate the element ID corresponding to a label.
 *
 * If the label isn't among the materials, a null ID will be returned.
 */
ElementId MaterialParams::find_element(std::string const& name) const
{
    auto result = el_labels_.find_all(name);
    if (result.empty())
        return {};
    CELER_VALIDATE(result.size() == 1,
                   << "element '" << name << "' is not unique");
    return result.front();
}

//---------------------------------------------------------------------------//
/*!
 * Get zero or more element IDs corresponding to a name.
 */
auto MaterialParams::find_elements(std::string const& name) const
    -> SpanConstElementId
{
    return el_labels_.find_all(name);
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
void MaterialParams::append_element_def(ElementInput const& inp,
                                        HostValue* host_data)
{
    CELER_EXPECT(inp.atomic_number);
    CELER_EXPECT(inp.atomic_mass > zero_quantity());

    ElementRecord result;

    // Copy basic properties
    result.atomic_number = inp.atomic_number;
    result.atomic_mass = inp.atomic_mass;

    // Calculate various factors of the atomic number
    const real_type z_real = result.atomic_number.unchecked_get();
    result.cbrt_z = std::cbrt(z_real);
    result.cbrt_zzp = std::cbrt(z_real * (z_real + 1));
    result.log_z = std::log(z_real);
    result.coulomb_correction
        = detail::calc_coulomb_correction(result.atomic_number);
    result.mass_radiation_coeff = detail::calc_mass_rad_coeff(result);

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
MaterialParams::extend_elcomponents(MaterialInput const& inp,
                                    HostValue* host_data) const
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
        components[i].element = inp.elements_fractions[i].first;
        components[i].fraction = inp.elements_fractions[i].second;
        // Add fractions to verify unity
        norm += inp.elements_fractions[i].second;
    }

    // Renormalize component fractions that are not unity and log them
    if (!inp.elements_fractions.empty() && !soft_equal(norm, 1.0))
    {
        CELER_LOG(warning) << "Element component fractions for `" << inp.label
                           << "` should sum to 1 but instead sum to " << norm
                           << " (difference = " << norm - 1 << ")";

        // Normalize
        norm = 1.0 / norm;
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
        [](MatElementComponent const& lhs, MatElementComponent const& rhs) {
            return lhs.element < rhs.element;
        });

    return make_builder(&host_data->elcomponents)
        .insert_back(components.begin(), components.end());
}

//---------------------------------------------------------------------------//
/*!
 * Convert an material input to an material definition and store.
 */
void MaterialParams::append_material_def(MaterialInput const& inp,
                                         HostValue* host_data)
{
    CELER_EXPECT(inp.number_density >= 0);
    CELER_EXPECT((inp.number_density == 0) == inp.elements_fractions.empty());
    CELER_EXPECT(host_data);

    MaterialRecord result;
    // Copy basic properties
    result.number_density = inp.number_density;
    result.temperature = inp.temperature;
    result.matter_state = inp.matter_state;
    result.elements = this->extend_elcomponents(inp, host_data);

    /*!
     * Calculate derived quantities: density, electron density, and rad length
     *
     * NOTE: Electron density calculation may need to be updated for solids.
     */
    double avg_amu_mass = 0;
    double avg_z = 0;
    double rad_coeff = 0;
    double log_mean_exc_energy = 0;
    for (MatElementComponent const& comp :
         host_data->elcomponents[result.elements])
    {
        CELER_ASSERT(comp.element < host_data->elements.size());
        ElementRecord const& el = host_data->elements[comp.element];
        real_type frac_z = comp.fraction * el.atomic_number.unchecked_get();

        avg_amu_mass += comp.fraction * el.atomic_mass.value();
        avg_z += frac_z;
        rad_coeff += comp.fraction * el.mass_radiation_coeff;
        log_mean_exc_energy
            += frac_z
               * std::log(value_as<units::MevEnergy>(
                   detail::get_mean_excitation_energy(el.atomic_number)));
    }
    result.zeff = avg_z;
    result.density = result.number_density * avg_amu_mass
                     * constants::atomic_mass;
    result.electron_density = result.number_density * avg_z;
    result.rad_length = 1 / (rad_coeff * result.density);
    log_mean_exc_energy = avg_z > 0 ? log_mean_exc_energy / avg_z
                                    : -numeric_limits<real_type>::infinity();
    result.log_mean_exc_energy = units::LogMevEnergy{log_mean_exc_energy};
    result.mean_exc_energy = units::MevEnergy{std::exp(log_mean_exc_energy)};

    // Add to host vector
    make_builder(&host_data->materials).push_back(result);

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
}  // namespace celeritas
