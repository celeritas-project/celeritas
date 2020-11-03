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
#include "base/Range.hh"
#include "base/SoftEqual.hh"
#include "base/SpanRemapper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a vector of material definitions.
 */
MaterialParams::MaterialParams(const Input& inp) : max_el_(0)
{
    REQUIRE(!inp.materials.empty());

    // Reserve host space (MUST reserve elcomponents to avoid invalidating
    // spans).
    host_elements_.reserve(inp.elements.size());
    host_elcomponents_.reserve(
        std::accumulate(inp.materials.begin(),
                        inp.materials.end(),
                        size_type(0),
                        [](size_type count, const MaterialInput& mi) {
                            return count + mi.elements.size();
                        }));
    host_materials_.reserve(inp.materials.size());
    elnames_.reserve(inp.elements.size());
    matnames_.reserve(inp.materials.size());

    // Build elements and materials on host.
    for (const auto& el : inp.elements)
    {
        this->append_element_def(el);
    }
    for (const auto& mat : inp.materials)
    {
        this->append_material_def(mat);
    }

#if CELERITAS_USE_CUDA
    // Allocate device vectors
    device_elements_ = DeviceVector<ElementDef>{host_elements_.size()};
    device_elcomponents_
        = DeviceVector<MatElementComponent>{host_elcomponents_.size()};
    device_materials_ = DeviceVector<MaterialDef>{host_materials_.size()};

    // Remap material->elcomponent spans
    std::vector<MaterialDef> temp_device_mats = host_materials_;
    auto                     remap_elements   = make_span_remapper(
        make_span(host_elcomponents_), device_elcomponents_.device_pointers());
    for (MaterialDef& m : temp_device_mats)
    {
        m.elements = remap_elements(m.elements);
    }

    // Copy vectors to device
    device_elements_.copy_to_device(make_span(host_elements_));
    device_elcomponents_.copy_to_device(make_span(host_elcomponents_));
    device_materials_.copy_to_device(make_span(temp_device_mats));
#endif

    ENSURE(host_elements_.size() == inp.elements.size());
    ENSURE(host_elcomponents_.size() <= host_elcomponents_.capacity());
    ENSURE(host_materials_.size() == inp.materials.size());
    ENSURE(elnames_.size() == inp.elements.size());
    ENSURE(matnames_.size() == inp.materials.size());
    ENSURE(matname_to_id_.size() == inp.materials.size());
}

//---------------------------------------------------------------------------//
/*!
 * Access material properties on the host.
 */
MaterialParamsPointers MaterialParams::host_pointers() const
{
    MaterialParamsPointers result;
    result.elements  = make_span(host_elements_);
    result.materials = make_span(host_materials_);
    ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//

/*!
 * Access material properties on the device.
 */
MaterialParamsPointers MaterialParams::device_pointers() const
{
    MaterialParamsPointers result;
    result.elements  = device_elements_.device_pointers();
    result.materials = device_materials_.device_pointers();
    ENSURE(result);
    return result;
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
void MaterialParams::append_element_def(const ElementInput& inp)
{
    REQUIRE(inp.atomic_number > 0);
    REQUIRE(inp.atomic_mass > zero_quantity());

    ElementDef result;

    // Copy basic properties
    result.atomic_number = inp.atomic_number;
    result.atomic_mass   = inp.atomic_mass;

    // Calculate various factors of the atomic number
    const real_type z_real      = result.atomic_number;
    result.cbrt_z               = std::cbrt(z_real);
    result.cbrt_zzp             = std::cbrt(z_real * (z_real + 1));
    result.log_z                = std::log(z_real);
    result.mass_radiation_coeff = detail::calc_mass_rad_coeff(result);

    elnames_.push_back(inp.name);

    // Add to host vector
    host_elements_.push_back(result);
}

//---------------------------------------------------------------------------//
/*!
 * Process and store element components to the internal list.
 *
 * \todo It's the caller's responsibility to ensure that element IDs
 * aren't duplicated.
 */
span<MatElementComponent>
MaterialParams::extend_elcomponents(const MaterialInput& inp)
{
    REQUIRE(host_elcomponents_.size() + inp.elements.size()
            <= host_elcomponents_.capacity());

    // Allocate material components
    auto start_size = host_elcomponents_.size();
    host_elcomponents_.resize(start_size + inp.elements.size());
    span<MatElementComponent> result{host_elcomponents_.data() + start_size,
                                     inp.elements.size()};

    // Copy values from input and check number fractions
    real_type norm = 0.0;
    for (auto i : range(inp.elements.size()))
    {
        REQUIRE(inp.elements[i].first < host_elements_.size());
        REQUIRE(inp.elements[i].second >= 0);
        result[i].element  = inp.elements[i].first;
        result[i].fraction = inp.elements[i].second;
        norm += inp.elements[i].second;
    }
    CHECK(SoftEqual<>()(norm, 1.0) || inp.elements.empty());
    norm = 1 / norm;

    // Normalize
    for (MatElementComponent& comp : result)
    {
        comp.fraction *= norm;
    }

    // Sort elements by increasing element ID for improved access
    std::sort(
        result.begin(),
        result.end(),
        [](const MatElementComponent& lhs, const MatElementComponent& rhs) {
            return lhs.element < rhs.element;
        });

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Convert an material input to an material definition and store.
 */
void MaterialParams::append_material_def(const MaterialInput& inp)
{
    REQUIRE(inp.number_density >= 0);
    REQUIRE((inp.number_density == 0) == inp.elements.empty());
    REQUIRE(inp.temperature > 0 || inp.number_density == 0);

    auto insertion = matname_to_id_.insert(
        {inp.name, MaterialDefId(host_materials_.size())});
    CHECK(insertion.second);

    MaterialDef result;

    // Copy basic properties
    result.number_density = inp.number_density;
    result.temperature    = inp.temperature;
    result.matter_state   = inp.matter_state;

    // Set elemental fractions
    result.elements = this->extend_elcomponents(inp);

    // Renormalize component fractions and calculate weighted properties
    real_type avg_amu   = 0;
    real_type avg_z     = 0;
    real_type rad_coeff = 0;
    for (const MatElementComponent& comp : result.elements)
    {
        CHECK(comp.element < host_elements_.size());
        const ElementDef& el = host_elements_[comp.element.get()];
        avg_amu += comp.fraction * el.atomic_mass.value();
        avg_z += comp.fraction * el.atomic_number;
        rad_coeff += comp.fraction * el.mass_radiation_coeff;
    }
    result.density = avg_amu * constants::atomic_mass * result.number_density;
    result.electron_density = avg_z * result.number_density;
    result.rad_length       = 1 / (rad_coeff * result.density);

    // Add to host vector
    host_materials_.push_back(result);
    matnames_.push_back(inp.name);

    // Update maximum number of materials
    max_el_ = std::max(max_el_, result.elements.size());

    ENSURE(result.number_density >= 0);
    ENSURE(result.temperature >= 0);
    ENSURE((result.density > 0) == (inp.number_density > 0));
    ENSURE((result.electron_density > 0) == (inp.number_density > 0));
    ENSURE(result.rad_length > 0);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
