//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MockImportedData.cc
//---------------------------------------------------------------------------//
#include "MockImportedData.hh"

#include "TestMacros.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//
struct MeterCubedPerMeV
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return ipow<3>(units::meter) / units::Mev::value();
    }

    static char const* label() { return "m^3/MeV"; }
};

using IsothermalCompressibility = Quantity<MeterCubedPerMeV>;

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
 *  j = 3,4: size = 3
 */
std::vector<ImportPhysicsVector>
mock_vec(std::vector<unsigned int> const& grid_indices,
         std::vector<unsigned int> const& value_indices)
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

    std::vector<std::vector<double>> mock_grids;
    for (auto i : grid_indices)
    {
        CELER_ASSERT(i < grids.size());
        mock_grids.push_back(grids[i]);
    }

    std::vector<std::vector<double>> mock_values;
    for (auto j : value_indices)
    {
        CELER_ASSERT(j < values.size());
        mock_values.push_back(values[j]);
    }

    return detail::convert_vector_units<units::Mev, units::Native>(
        mock_grids, mock_values);
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
        {IMC::absorption, mock_vec({0, 1, 1, 2, 3}, {0, 1, 2, 3, 4})},
        {IMC::rayleigh, mock_vec({1, 0, 3, 2, 0}, {0, 1, 3, 3, 2})},
        {IMC::wls, mock_vec({3, 1, 1, 2, 3}, {4, 0, 1, 4, 4})}};

    return models;
}

//---------------------------------------------------------------------------//
/*!
 * Construct vector of ImportOpticalMaterial from mock data.
 */
std::vector<ImportOpticalMaterial> const& MockImportedData::import_materials()
{
    using namespace celeritas::units;

    static std::vector<std::vector<double>> mock_energies
        = {{1.098177, 1.256172, 1.484130}, {1.098177, 6.812319}, {1, 2, 5}};

    static std::vector<std::vector<double>> mock_rindex
        = {{1.3235601610672, 1.3256740639273, 1.3280120256415},
           {1.3235601610672, 1.4679465862259},
           {1.3, 1.4, 1.5}};

    auto properties
        = detail::convert_vector_units<detail::ElectronVolt, units::Native>(
            mock_energies, mock_rindex);

    static ImportOpticalRayleigh mock_rayleigh[]
        = {{1, 7.658e-23 * MeterCubedPerMeV::value()},
           {1.7, 4.213e-24 * MeterCubedPerMeV::value()},
           {2, 1e-20 * MeterCubedPerMeV::value()}};

    static std::vector<ImportOpticalMaterial> materials{
        ImportOpticalMaterial{ImportOpticalProperty{properties[0]},
                              ImportScintData{},
                              ImportOpticalRayleigh{mock_rayleigh[0]},
                              ImportWavelengthShift{}},
        ImportOpticalMaterial{ImportOpticalProperty{properties[0]},
                              ImportScintData{},
                              ImportOpticalRayleigh{mock_rayleigh[1]},
                              ImportWavelengthShift{}},
        ImportOpticalMaterial{ImportOpticalProperty{properties[1]},
                              ImportScintData{},
                              ImportOpticalRayleigh{mock_rayleigh[0]},
                              ImportWavelengthShift{}},
        ImportOpticalMaterial{ImportOpticalProperty{properties[2]},
                              ImportScintData{},
                              ImportOpticalRayleigh{mock_rayleigh[2]},
                              ImportWavelengthShift{}},
        ImportOpticalMaterial{ImportOpticalProperty{properties[1]},
                              ImportScintData{},
                              ImportOpticalRayleigh{mock_rayleigh[1]},
                              ImportWavelengthShift{}}};

    return materials;
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve optical materials constructed from mock imported data.
 *
 * Will only construct the materials when called, and only once.
 */
auto MockImportedData::optical_materials() const -> SPConstMaterials const&
{
    static SPConstMaterials materials = nullptr;
    if (!materials)
    {
        materials = this->build_optical_materials();
    }
    return materials;
}

//---------------------------------------------------------------------------//
/*!
 * Build optical material parameters from imported data.
 */
auto MockImportedData::build_optical_materials() const -> SPConstMaterials
{
    MaterialParams::Input input;
    for (auto mat : MockImportedData::import_materials())
    {
        input.properties.push_back(mat.properties);
    }

    // Volume -> optical material mapping with some redundancies
    for (auto opt_mat : range(8))
    {
        input.volume_to_mat.push_back(
            OpticalMaterialId(opt_mat % input.properties.size()));
    }

    return std::make_shared<MaterialParams const>(std::move(input));
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
    ImportPhysicsVector const empty_vec{ImportPhysicsVectorType::free, {}, {}};
    for (auto const& model : this->import_models())
    {
        empty_models.push_back(
            {model.model_class,
             ImportedMfpTable(model.mfp_table.size(), empty_vec)});
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
                                         ItemRange<Grid> const& table,
                                         bool soft) const
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
        if (soft)
        {
            EXPECT_VEC_SOFT_EQ(expected_grid, reals[grid.grid]);
            EXPECT_VEC_SOFT_EQ(expected_value, reals[grid.value]);
        }
        else
        {
            EXPECT_VEC_EQ(expected_grid, reals[grid.grid]);
            EXPECT_VEC_EQ(expected_value, reals[grid.value]);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Check the built physics table with soft equality.
 */
void MockImportedData::check_built_table_soft(ImportedMfpTable const& expected,
                                              ItemRange<Grid> const& table) const
{
    this->check_built_table(expected, table, true);
}

//---------------------------------------------------------------------------//
/*!
 * Check the built physics table with exact equality.
 */
void MockImportedData::check_built_table_exact(
    ImportedMfpTable const& expected, ItemRange<Grid> const& table) const
{
    this->check_built_table(expected, table, false);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
