//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalMockTestBase.cc
//---------------------------------------------------------------------------//
#include "OpticalMockTestBase.hh"

#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/io/ImportOpticalModel.hh"

#include "ValidationUtils.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//
// UNITS
//---------------------------------------------------------------------------//
using TimeSecond = celeritas::RealQuantity<celeritas::units::Second>;

struct Kelvin
{
    static CELER_CONSTEXPR_FUNCTION Constant value() { return units::kelvin; }

    static char const* label() { return "K"; }
};

struct MeterCubedPerMeV
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return ipow<3>(units::meter) / units::Mev::value();
    }

    static char const* label() { return "m^3/MeV"; }
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Helper function for converting hardcoded grids into \c inp::Grid.
 *
 * The grid energy is converted to units of MeV, while the values are converted
 * to native units.
 */
template<class GridUnit, class ValueUnit>
inp::Grid
native_physics_vector_from(std::vector<double> xs, std::vector<double> ys)
{
    CELER_EXPECT(xs.size() == ys.size());
    inp::Grid v{std::move(xs), std::move(ys), inp::Interpolation{}};
    for (double& x : v.x)
    {
        x = value_as<units::MevEnergy>(native_value_to<units::MevEnergy>(
            native_value_from(RealQuantity<GridUnit>(x))));
    }

    for (double& y : v.y)
    {
        y = native_value_from(RealQuantity<ValueUnit>(y));
    }

    return v;
}

//---------------------------------------------------------------------------//
/*!
 * Helper function for converting hardcoded tables (lists of grids) into
 * \c inp::Grid.
 */
template<class GridUnit, class ValueUnit>
std::vector<inp::Grid> native_physics_table_from(
    std::vector<std::tuple<std::vector<double>, std::vector<double>>> data)
{
    std::vector<inp::Grid> table;
    table.reserve(data.size());
    for (auto&& arrs : data)
    {
        table.push_back(native_physics_vector_from<GridUnit, ValueUnit>(
            std::move(std::get<0>(arrs)), std::move(std::get<1>(arrs))));
    }

    return table;
}

//---------------------------------------------------------------------------//
// OpticalMockTestBase
//---------------------------------------------------------------------------//
/*!
 * Constructs optical material parameters from mock data.
 */
auto OpticalMockTestBase::build_optical_material() -> SPConstOpticalMaterial
{
    MaterialParams::Input input;
    for (auto mat : this->imported_data().optical_materials)
    {
        input.properties.push_back(mat.properties);
    }

    // Volume -> optical material mapping with some redundancies
    for (auto opt_mat : range(8))
    {
        input.volume_to_mat.push_back(
            OptMatId(opt_mat % input.properties.size()));
    }

    // mock PhysMatId == OptMatId
    for (auto mat : range(PhysMatId(input.properties.size())))
    {
        input.optical_to_core.push_back(mat);
    }

    return std::make_shared<MaterialParams const>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Constructs (core) material parameters from mock data.
 *
 * Only temperatures and optical material IDs are assigned meaningful values.
 */
auto OpticalMockTestBase::build_material() -> SPConstMaterial
{
    ::celeritas::MaterialParams::Input input;

    static constexpr auto material_temperatures
        = native_array_from<RealQuantity<Kelvin>>(
            283.15, 300.0, 283.15, 200., 300.0);

    // Unused element - only to pass checks
    input.elements.push_back(::celeritas::MaterialParams::ElementInput{
        AtomicNumber{1}, units::AmuMass{1}, {}, "fake"});

    for (auto i : range(material_temperatures.size()))
    {
        // Only temperature is relevant information
        input.materials.push_back(::celeritas::MaterialParams::MaterialInput{
            0,
            material_temperatures[i],
            MatterState::solid,
            {},
            std::to_string(i).c_str()});

        // mock PhysMatId == OptMatId
        input.mat_to_optical.push_back(OptMatId(i));
    }

    return std::make_shared<::celeritas::MaterialParams const>(
        std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Access mock imported data.
 */
ImportData const& OpticalMockTestBase::imported_data() const
{
    static ImportData data;
    if (data.optical_materials.empty())
    {
        this->build_import_data(data);
    }
    return data;
}

//---------------------------------------------------------------------------//
/*!
 * Create mock imported data in-place.
 */
void OpticalMockTestBase::build_import_data(ImportData& data) const
{
    data.units = units::NativeTraits::label();
    using Compressibility = RealQuantity<MeterCubedPerMeV>;

    // Build mock imported optical materials
    {
        data.optical_materials.resize(5);

        data.optical_materials[0].properties.refractive_index
            = native_physics_vector_from<units::ElectronVolt, units::Native>(
                {1.098177, 1.256172, 1.484130},
                {1.3235601610672, 1.3256740639273, 1.3280120256415});
        data.optical_materials[0].rayleigh.scale_factor = 1;
        data.optical_materials[0].rayleigh.compressibility
            = native_value_from(Compressibility{7.658e-23});
        data.optical_materials[0].wls.mean_num_photons = 2;
        data.optical_materials[0].wls.time_constant
            = native_value_from(TimeSecond(1e-9));
        // Reemitted photon energy range (visible light)
        data.optical_materials[0].wls.component.x
            = {1.65e-6, 2e-6, 2.4e-6, 2.8e-6, 3.26e-6};
        // Reemitted photon energy spectrum
        data.optical_materials[0].wls.component.y
            = {0.15, 0.25, 0.50, 0.40, 0.02};
        data.optical_materials[0].wls2.mean_num_photons = 1;
        data.optical_materials[0].wls2.time_constant
            = native_value_from(TimeSecond(21.7e-9));
        // Reemitted photon energy range (visible light)
        data.optical_materials[0].wls2.component.x
            = {1.65e-6, 2e-6, 2.4e-6, 2.8e-6, 3.26e-6};
        // Reemitted photon energy spectrum
        data.optical_materials[0].wls2.component.y
            = {0.016, 0.024, 0.040, 0.111, 0.206, 0.325, 0.413};
        data.optical_materials[0].wls2.component.x = {
            1.771e-6, 1.850e-6, 1.901e-6, 2.003e-6, 2.073e-6, 2.141e-6, 2.171e-6};

        data.optical_materials[1].properties.refractive_index
            = native_physics_vector_from<units::ElectronVolt, units::Native>(
                {1.098177, 1.256172, 1.484130},
                {1.3235601610672, 1.3256740639273, 1.3280120256415});
        data.optical_materials[1].rayleigh.scale_factor = 1.7;
        data.optical_materials[1].rayleigh.compressibility
            = native_value_from(Compressibility{4.213e-24});

        data.optical_materials[2].properties.refractive_index
            = native_physics_vector_from<units::ElectronVolt, units::Native>(
                {1.098177, 6.812319}, {1.3235601610672, 1.4679465862259});
        data.optical_materials[2].rayleigh.scale_factor = 1;
        data.optical_materials[2].rayleigh.compressibility
            = native_value_from(Compressibility{7.658e-23});

        data.optical_materials[3].properties.refractive_index
            = native_physics_vector_from<units::ElectronVolt, units::Native>(
                {1, 2, 5}, {1.3, 1.4, 1.5});
        data.optical_materials[3].rayleigh.scale_factor = 2;
        data.optical_materials[3].rayleigh.compressibility
            = native_value_from(Compressibility{1e-20});

        data.optical_materials[4].properties.refractive_index
            = native_physics_vector_from<units::ElectronVolt, units::Native>(
                {1.098177, 6.812319}, {1.3235601610672, 1.4679465862259});
        data.optical_materials[4].rayleigh.scale_factor = 1.7;
        data.optical_materials[4].rayleigh.compressibility
            = native_value_from(Compressibility{4.213e-24});
    }

    // Build mock imported optical models
    {
        data.optical_models.resize(4);

        data.optical_models[0].model_class = ImportModelClass::absorption;
        data.optical_models[0].mfp_table
            = native_physics_table_from<units::Mev, units::Centimeter>({
                {{1e-3, 1e-2}, {5.7, 9.3}},
                {{1e-2, 3e2}, {1.2, 10.7}},
                {{1e-2, 3e2}, {3.1, 5.4}},
                {{2e-3, 5e1, 1e2}, {0.1, 7.6, 12.5}},
                {{1e-3, 2e-3, 5e-1}, {1.3, 4.9, 9.4}},
            });

        data.optical_models[1].model_class = ImportModelClass::rayleigh;
        data.optical_models[1].mfp_table
            = native_physics_table_from<units::Mev, units::Centimeter>({
                {{1e-2, 3e2}, {5.7, 9.3}},
                {{1e-3, 1e-2}, {1.2, 10.7}},
                {{1e-3, 2e-3, 5e-1}, {0.1, 7.6, 12.5}},
                {{2e-3, 5e1, 1e2}, {0.1, 7.6, 12.5}},
                {{1e-3, 1e-2}, {3.1, 5.4}},
            });

        data.optical_models[2].model_class = ImportModelClass::wls;
        data.optical_models[2].mfp_table
            = native_physics_table_from<units::Mev, units::Centimeter>({
                {{1e-3, 2e-3, 5e-1}, {1.3, 4.9, 9.4}},
                {{1e-2, 3e2}, {5.7, 9.3}},
                {{1e-2, 3e2}, {1.2, 10.7}},
                {{2e-3, 5e1, 1e2}, {1.3, 4.9, 9.4}},
                {{1e-3, 2e-3, 5e-1}, {1.3, 4.9, 9.4}},
            });

        data.optical_models[3].model_class = ImportModelClass::wls2;
        data.optical_models[3].mfp_table
            = native_physics_table_from<units::Mev, units::Centimeter>({
                {{1e-1, 1e1}, {2.3, 5.4}},
                {{2e-2, 1e0, 3e2}, {5.7, 6.2, 9.3}},
                {{3e-2, 3e2}, {3.2, 9.4}},
                {{2e-3, 2e2}, {4.9, 9.4}},
                {{1e-3, 4e-3, 5e-1}, {1.3, 5.9, 8.4}},
            });
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get the imported optical model corresponding to the given \c
 * ImportModelClass.
 */
ImportOpticalModel const&
OpticalMockTestBase::import_model_by_class(ImportModelClass imc) const
{
    switch (imc)
    {
        case ImportModelClass::absorption:
            return this->imported_data().optical_models[0];
        case ImportModelClass::rayleigh:
            return this->imported_data().optical_models[1];
        case ImportModelClass::wls:
            return this->imported_data().optical_models[2];
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
