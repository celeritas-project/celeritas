//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MockImportedData.cc
//---------------------------------------------------------------------------//
#include "MockImportedData.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Convert vector of some floating type to a vector of reals.
 */
template<typename T>
std::vector<real_type> convert_to_reals(std::vector<T> const& xs)
{
    std::vector<real_type> reals;
    reals.reserve(xs.size());
    for (T x : xs)
    {
        reals.push_back(static_cast<real_type>(x));
    }
    return reals;
}

//---------------------------------------------------------------------------//
/*!
 * Create some mock physics vectors.
 *
 * For x grid:
 *  i = 0,1: size = 2
 *  i = 2,3: size = 3
 *
 * For y values:
 *  j = 0,1,2: size = 2
 *  j = 3,4: sie = 3
 */
ImportPhysicsVector mock_vec(unsigned int i, unsigned int j)
{
    static std::vector<std::vector<double>> grids{
        {1e-3, 1e2}, {1e-2, 3e2}, {2e-3, 5e-1, 1e2}, {1e-3, 2e-3, 5e-1}};

    static std::vector<std::vector<double>> values{
        {5.7, 9.3},
        {1.2, 10.7},
        {3.1, 5.4},
        {0.1, 7.6, 12.5},
        {1.3, 4.9, 9.4},
    };

    CELER_EXPECT(i < grids.size());
    CELER_EXPECT(j < values.size());
    CELER_EXPECT(grids[i].size() == values[j].size());

    return ImportPhysicsVector{
        ImportPhysicsVectorType::linear, grids[i], values[j]};
}

//---------------------------------------------------------------------------//
/*!
 * Construct vector of ImportOpticalModel from mock data.
 *
 * There are 4 imported models, one for each optical model class. All models
 * have MFP grids for 5 materials.
 */
std::vector<ImportOpticalModel> const& MockImportedData::import_models()
{
    using IMC = ImportModelClass;

    static std::vector<ImportOpticalModel> models{
        {IMC::absorption,
         {mock_vec(0, 0),
          mock_vec(1, 1),
          mock_vec(1, 2),
          mock_vec(2, 3),
          mock_vec(3, 4)}},
        {IMC::rayleigh,
         {mock_vec(1, 0),
          mock_vec(0, 1),
          mock_vec(3, 3),
          mock_vec(2, 3),
          mock_vec(0, 2)}},
        {IMC::wls,
         {mock_vec(3, 4),
          mock_vec(1, 0),
          mock_vec(1, 1),
          mock_vec(2, 4),
          mock_vec(3, 4)}},
        {IMC::other,
         {mock_vec(0, 0),
          mock_vec(1, 0),
          mock_vec(2, 4),
          mock_vec(3, 3),
          mock_vec(1, 0)}},
    };

    return models;
}

//---------------------------------------------------------------------------//
/*!
 * Construct vector of ImportOpticalMaterial from mock data.
 *
 * Currently returns a vector of 5 empty materials.
 * TODO: Reimplement with mock data for models that depend on material data.
 */
std::vector<ImportOpticalMaterial> const& MockImportedData::import_materials()
{
    static std::vector<ImportOpticalMaterial> materials(
        5, ImportOpticalMaterial());
    return materials;
}

//---------------------------------------------------------------------------//
/*!
 * Imported model ID for absorption.
 */
auto MockImportedData::absorption_id() -> ImportedModelId
{
    return ImportedModelId{0};
}

//---------------------------------------------------------------------------//
/*!
 * Imported model ID for Rayleigh scattering.
 */
auto MockImportedData::rayleigh_id() -> ImportedModelId
{
    return ImportedModelId{1};
}

//---------------------------------------------------------------------------//
/*!
 * Imported model ID for wavelength shifting.
 */
auto MockImportedData::wls_id() -> ImportedModelId
{
    return ImportedModelId{2};
}

//---------------------------------------------------------------------------//
/*!
 * Create ImportedModels all with empty MFP grids.
 *
 * Useful for testing optical models which build their MFPs from material data.
 */
auto MockImportedData::create_empty_imported_models() const -> SPConstImported
{
    std::vector<ImportOpticalModel> empty_models;
    empty_models.reserve(this->import_models().size());
    ImportPhysicsVector const empty_vec{
        ImportPhysicsVectorType::linear, {}, {}};
    for (auto const& model : this->import_models())
    {
        empty_models.push_back(
            {model.model_class, ImportedMfpTable(model.mfps.size(), empty_vec)});
    }

    return std::make_shared<ImportedModels const>(std::move(empty_models));
}

//---------------------------------------------------------------------------//
/*!
 * Create ImportedModels from mock data.
 */
auto MockImportedData::create_imported_models() const -> SPConstImported
{
    return std::make_shared<ImportedModels const>(this->import_models());
}

//---------------------------------------------------------------------------//
/*!
 * Create an MFP builder that uses this object's collections.
 */
auto MockImportedData::create_mfp_builder() -> MfpBuilder
{
    return MfpBuilder(&reals, &grids);
}

//---------------------------------------------------------------------------//
/*!
 * Check that two MFP physics vectors are equal.
 */
void MockImportedData::check_mfp(ImportPhysicsVector const& expected,
                                 ImportPhysicsVector const& imported) const
{
    EXPECT_EQ(expected.vector_type, imported.vector_type);
    EXPECT_VEC_EQ(expected.x, imported.x);
    EXPECT_VEC_EQ(expected.y, imported.y);
}

//---------------------------------------------------------------------------//
/*!
 * Check that the physics table built in the collections matches the
 * imported MFP table it was built from.
 */
void MockImportedData::check_built_table(ImportedMfpTable const& expected_mfps,
                                         ItemRange<Grid> const& table) const
{
    // Each MFP has a built grid
    ASSERT_EQ(expected_mfps.size(), table.size());

    for (auto mfp_id : range(expected_mfps.size()))
    {
        // Grid IDs should be valid
        auto grid_id = table[mfp_id];
        ASSERT_LT(grid_id, grids.size());

        // Grid should be valid
        Grid const& grid = grids[grid_id];
        ASSERT_TRUE(grid);

        // Grid ranges should be valid
        ASSERT_LT(grid.grid.back(), reals.size());
        ASSERT_LT(grid.value.back(), reals.size());

        // Convert imported data to real_type for comparison
        auto const& expected_mfp = expected_mfps[mfp_id];
        std::vector<real_type> expected_grid = convert_to_reals(expected_mfp.x);
        std::vector<real_type> expected_value
            = convert_to_reals(expected_mfp.y);

        // Built grid data should match expected grid data
        EXPECT_VEC_EQ(expected_grid, reals[grid.grid]);
        EXPECT_VEC_EQ(expected_value, reals[grid.value]);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
