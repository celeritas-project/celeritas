//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalMockTestBase.cc
//---------------------------------------------------------------------------//
#include "OpticalMockTestBase.hh"

#include "corecel/Assert.hh"
#include "celeritas/UnitTypes.hh"
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
    static ImportData const data = [this] {
        ImportData d;
        this->build_import_data(d);
        return d;
    }();
    return data;
}

//---------------------------------------------------------------------------//
/*!
 * Create mock imported data in-place.
 */
void OpticalMockTestBase::build_import_data(ImportData& data) const
{
    data.units = units::NativeTraits::label();
    auto& bulk = data.optical_physics.bulk;
    using Compressibility = RealQuantity<MeterCubedPerMeV>;

    auto mat_props = [&data](std::size_t opt_mat_idx) -> ImportOpticalProperty& {
        if (opt_mat_idx >= data.optical_materials.size())
            data.optical_materials.resize(opt_mat_idx + 1);
        return data.optical_materials[opt_mat_idx].properties;
    };

    auto model = [](std::size_t opt_mat_idx, auto&& obm) -> auto& {
        // NOTE: parens required to return a reference
        return (obm.materials[OptMatId(opt_mat_idx)]);
    };
    auto set_mfp = [&](std::size_t mat_idx,
                       auto&& obm,
                       std::pair<std::vector<double>, std::vector<double>> xy) {
        model(mat_idx, obm).mfp
            = native_physics_vector_from<units::Mev, units::Centimeter>(
                std::move(xy.first), std::move(xy.second));
    };

    // Material 0
    mat_props(0).refractive_index
        = native_physics_vector_from<units::ElectronVolt, units::Native>(
            {1.098177, 1.256172, 1.484130},
            {1.3235601610672, 1.3256740639273, 1.3280120256415});
    model(0, bulk.rayleigh).scale_factor = 1;
    model(0, bulk.rayleigh).compressibility
        = native_value_from(Compressibility{7.658e-23});
    model(0, bulk.wls).mean_num_photons = 2;
    model(0, bulk.wls).time_constant = native_value_from(TimeSecond(1e-9));
    model(0, bulk.wls).component.x = {1.65e-6, 2e-6, 2.4e-6, 2.8e-6, 3.26e-6};
    model(0, bulk.wls).component.y = {0.15, 0.25, 0.50, 0.40, 0.02};
    model(0, bulk.wls2).mean_num_photons = 1;
    model(0, bulk.wls2).time_constant = native_value_from(TimeSecond(21.7e-9));
    model(0, bulk.wls2).component.x = {
        1.771e-6, 1.850e-6, 1.901e-6, 2.003e-6, 2.073e-6, 2.141e-6, 2.171e-6};
    model(0, bulk.wls2).component.y
        = {0.016, 0.024, 0.040, 0.111, 0.206, 0.325, 0.413};
    model(0, bulk.mie).forward_g = 0.99;
    model(0, bulk.mie).backward_g = 0.99;
    model(0, bulk.mie).forward_ratio = 0.8;
    set_mfp(0, bulk.absorption, {{1e-3, 1e-2}, {5.7, 9.3}});
    set_mfp(0, bulk.rayleigh, {{1e-2, 3e2}, {5.7, 9.3}});
    set_mfp(0, bulk.wls, {{1e-3, 2e-3, 5e-1}, {1.3, 4.9, 9.4}});
    set_mfp(0, bulk.wls2, {{1e-1, 1e1}, {2.3, 5.4}});
    set_mfp(0, bulk.mie, {{1e-1, 1e1}, {2.3, 5.4}});

    // Material 1
    mat_props(1).refractive_index
        = native_physics_vector_from<units::ElectronVolt, units::Native>(
            {1.098177, 1.256172, 1.484130},
            {1.3235601610672, 1.3256740639273, 1.3280120256415});
    model(1, bulk.rayleigh).scale_factor = 1.7;
    model(1, bulk.rayleigh).compressibility
        = native_value_from(Compressibility{4.213e-24});
    set_mfp(1, bulk.absorption, {{1e-2, 3e2}, {1.2, 10.7}});
    set_mfp(1, bulk.rayleigh, {{1e-3, 1e-2}, {1.2, 10.7}});
    set_mfp(1, bulk.wls, {{1e-2, 3e2}, {5.7, 9.3}});
    set_mfp(1, bulk.wls2, {{2e-2, 1e0, 3e2}, {5.7, 6.2, 9.3}});
    set_mfp(1, bulk.mie, {{2e-2, 1e0, 3e2}, {5.7, 6.2, 9.3}});

    // Material 2
    mat_props(2).refractive_index
        = native_physics_vector_from<units::ElectronVolt, units::Native>(
            {1.098177, 6.812319}, {1.3235601610672, 1.4679465862259});
    model(2, bulk.rayleigh).scale_factor = 1;
    model(2, bulk.rayleigh).compressibility
        = native_value_from(Compressibility{7.658e-23});
    set_mfp(2, bulk.absorption, {{1e-2, 3e2}, {3.1, 5.4}});
    set_mfp(2, bulk.rayleigh, {{1e-3, 2e-3, 5e-1}, {0.1, 7.6, 12.5}});
    set_mfp(2, bulk.wls, {{1e-2, 3e2}, {1.2, 10.7}});
    set_mfp(2, bulk.wls2, {{3e-2, 3e2}, {3.2, 9.4}});
    set_mfp(2, bulk.mie, {{3e-2, 3e2}, {3.2, 9.4}});

    // Material 3
    mat_props(3).refractive_index
        = native_physics_vector_from<units::ElectronVolt, units::Native>(
            {1, 2, 5}, {1.3, 1.4, 1.5});
    model(3, bulk.rayleigh).scale_factor = 2;
    model(3, bulk.rayleigh).compressibility
        = native_value_from(Compressibility{1e-20});
    set_mfp(3, bulk.absorption, {{2e-3, 5e1, 1e2}, {0.1, 7.6, 12.5}});
    set_mfp(3, bulk.rayleigh, {{2e-3, 5e1, 1e2}, {0.1, 7.6, 12.5}});
    set_mfp(3, bulk.wls, {{2e-3, 5e1, 1e2}, {1.3, 4.9, 9.4}});
    set_mfp(3, bulk.wls2, {{2e-3, 2e2}, {4.9, 9.4}});
    set_mfp(3, bulk.mie, {{2e-3, 2e2}, {4.9, 9.4}});

    // Material 4
    mat_props(4).refractive_index
        = native_physics_vector_from<units::ElectronVolt, units::Native>(
            {1.098177, 6.812319}, {1.3235601610672, 1.4679465862259});
    model(4, bulk.rayleigh).scale_factor = 1.7;
    model(4, bulk.rayleigh).compressibility
        = native_value_from(Compressibility{4.213e-24});
    set_mfp(4, bulk.absorption, {{1e-3, 2e-3, 5e-1}, {1.3, 4.9, 9.4}});
    set_mfp(4, bulk.rayleigh, {{1e-3, 1e-2}, {3.1, 5.4}});
    set_mfp(4, bulk.wls, {{1e-3, 2e-3, 5e-1}, {1.3, 4.9, 9.4}});
    set_mfp(4, bulk.wls2, {{1e-3, 4e-3, 5e-1}, {1.3, 5.9, 8.4}});
    set_mfp(4, bulk.mie, {{1e-3, 4e-3, 5e-1}, {1.3, 5.9, 8.4}});
}

//---------------------------------------------------------------------------//
/*!
 * Get the imported optical model corresponding to the given \c
 * ImportModelClass.
 */
auto OpticalMockTestBase::get_mfp_table(ImportModelClass imc) const -> VecGrid
{
    VecGrid result;
    auto const& bulk = this->imported_data().optical_physics.bulk;
    auto make_table = [&](auto const& omb) {
        CELER_ASSERT(omb.model_class == imc);
        for (auto&& [id, omm] : omb.materials)
        {
            result.push_back(omm.mfp);
        }
    };
    switch (imc)
    {
        case ImportModelClass::absorption:
            make_table(bulk.absorption);
            break;
        case ImportModelClass::rayleigh:
            make_table(bulk.rayleigh);
            break;
        case ImportModelClass::wls:
            make_table(bulk.wls);
            break;
        case ImportModelClass::wls2:
            make_table(bulk.wls2);
            break;
        case ImportModelClass::mie:
            make_table(bulk.mie);
            break;
        default:
            CELER_ASSERT_UNREACHABLE();
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
