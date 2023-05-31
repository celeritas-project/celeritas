//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/MaterialParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/io/Label.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/AtomicNumber.hh"

#include "ElementView.hh"
#include "MaterialData.hh"
#include "MaterialView.hh"

namespace celeritas
{
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Data management for material, element, and nuclide properties.
 */
class MaterialParams final : public ParamsDataInterface<MaterialParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SpanConstMaterialId = Span<MaterialId const>;
    using SpanConstElementId = Span<ElementId const>;
    //!@}

    //! Define an element's input data
    struct ElementInput
    {
        AtomicNumber atomic_number;  //!< Z number
        units::AmuMass atomic_mass;  //!< Isotope-weighted average atomic mass
        Label label;  //!< Element name
    };

    //! Define a material's input data
    struct MaterialInput
    {
        real_type number_density;  //!< Atomic number density [1/cm^3]
        real_type temperature;  //!< Temperature [K]
        MatterState matter_state;  //!< Solid, liquid, gas
        std::vector<std::pair<ElementId, real_type>>
            elements_fractions;  //!< Fraction of number density
        Label label;  //!< Material name
    };

    //! Input data to construct this class
    struct Input
    {
        std::vector<ElementInput> elements;
        std::vector<MaterialInput> materials;
    };

  public:
    // Construct with imported data
    static std::shared_ptr<MaterialParams> from_import(ImportData const& data);

    // Construct with a vector of material definitions
    explicit MaterialParams(Input const& inp);

    //! Number of material definitions
    MaterialId::size_type size() const { return mat_labels_.size(); }

    //!@{
    //! \name Material metadata
    //! Number of materials
    MaterialId::size_type num_materials() const { return mat_labels_.size(); }

    // Get material name
    Label const& id_to_label(MaterialId id) const;

    // Find a material from a name
    MaterialId find_material(std::string const& name) const;

    // Find all materials that share a name
    SpanConstMaterialId find_materials(std::string const& name) const;
    //!@}

    //!@{
    //! \name Element metadata
    //! Number of distinct elements definitions
    ElementId::size_type num_elements() const { return el_labels_.size(); }

    // Get element name
    Label const& id_to_label(ElementId id) const;

    // Find an element from a name
    ElementId find_element(std::string const& name) const;

    // Find all elements that share a name
    SpanConstElementId find_elements(std::string const& name) const;
    //!@}

    // Access material definitions on host
    inline MaterialView get(MaterialId id) const;

    // Access element definitions on host
    inline ElementView get(ElementId id) const;

    // Maximum number of elements in any one material
    inline ElementComponentId::size_type max_element_components() const;

    //! Access material properties on the host
    HostRef const& host_ref() const final { return data_.host(); }

    //! Access material properties on the device
    DeviceRef const& device_ref() const final { return data_.device(); }

  private:
    // Metadata
    LabelIdMultiMap<MaterialId> mat_labels_;
    LabelIdMultiMap<ElementId> el_labels_;

    // Host/device storage and reference
    CollectionMirror<MaterialParamsData> data_;

    // HELPER FUNCTIONS
    using HostValue = HostVal<MaterialParamsData>;
    void append_element_def(ElementInput const& inp, HostValue*);
    ItemRange<MatElementComponent>
    extend_elcomponents(MaterialInput const& inp, HostValue*) const;
    void append_material_def(MaterialInput const& inp, HostValue*);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get material properties for the given material.
 */
MaterialView MaterialParams::get(MaterialId id) const
{
    CELER_EXPECT(id < this->host_ref().materials.size());
    return MaterialView(this->host_ref(), id);
}

//---------------------------------------------------------------------------//
/*!
 * Get properties for the given element.
 */
ElementView MaterialParams::get(ElementId id) const
{
    CELER_EXPECT(id < this->host_ref().elements.size());
    return ElementView(this->host_ref(), id);
}

//---------------------------------------------------------------------------//
/*!
 * Maximum number of elements in any one material.
 */
ElementComponentId::size_type MaterialParams::max_element_components() const
{
    return this->host_ref().max_element_components;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
