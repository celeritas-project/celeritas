//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialParamsLoader.cc
//---------------------------------------------------------------------------//
#include "MaterialParamsLoader.hh"

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>

#include "base/Assert.hh"
#include "ImportMaterial.hh"
#include "GdmlGeometryMap.hh"

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
 * Construct with RootLoader.
 */
MaterialParamsLoader::MaterialParamsLoader(RootLoader root_loader)
    : root_loader_(root_loader)
{
    CELER_ENSURE(root_loader);
}

//---------------------------------------------------------------------------//
/*!
 * Load MaterialParams data.
 */
const std::shared_ptr<const MaterialParams> MaterialParamsLoader::operator()()
{
    const auto tfile = root_loader_.get();

    // Open geometry branch
    std::unique_ptr<TTree> tree_geometry(tfile->Get<TTree>("geometry"));
    CELER_ASSERT(tree_geometry);
    CELER_ASSERT(tree_geometry->GetEntries() == 1);

    // Load branch and fetch data
    GdmlGeometryMap  geometry;
    GdmlGeometryMap* geometry_ptr = &geometry;

    int err_code
        = tree_geometry->SetBranchAddress("GdmlGeometryMap", &geometry_ptr);
    CELER_ASSERT(err_code >= 0);
    tree_geometry->GetEntry(0);

    // Create MaterialParams input for its constructor
    MaterialParams::Input input;

    // Populate input.elements
    for (const auto& elem_key : geometry.elemid_to_element_map())
    {
        MaterialParams::ElementInput element_params;
        element_params.atomic_number = elem_key.second.atomic_number;
        element_params.atomic_mass
            = units::AmuMass{elem_key.second.atomic_mass};
        element_params.name = elem_key.second.name;

        input.elements.push_back(element_params);
    }

    // Populate input.materials
    for (const auto& mat_key : geometry.matid_to_material_map())
    {
        MaterialParams::MaterialInput material_params;
        material_params.name           = mat_key.second.name;
        material_params.temperature    = mat_key.second.temperature;
        material_params.number_density = mat_key.second.number_density;
        material_params.matter_state   = to_matter_state(mat_key.second.state);

        for (const auto& elem_key : mat_key.second.elements_num_fractions)
        {
            ElementId elem_def_id{elem_key.first};

            // Populate MaterialParams number fractions
            material_params.elements_fractions.push_back(
                {elem_def_id, elem_key.second});
        }
        input.materials.push_back(material_params);
    }

    // Construct MaterialParams and return it as a shared_ptr
    MaterialParams materials(input);
    return std::make_shared<MaterialParams>(std::move(materials));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
