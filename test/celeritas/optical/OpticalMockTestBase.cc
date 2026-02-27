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
    using Compressibility = RealQuantity<MeterCubedPerMeV>;

    auto mat = [&data](std::size_t opt_mat_idx) -> ImportOpticalMaterial& {
        if (opt_mat_idx >= data.optical_materials.size())
            data.optical_materials.resize(opt_mat_idx + 1);
        return data.optical_materials[opt_mat_idx];
    };
    auto model = [&data](std::size_t model_idx) -> ImportOpticalModel& {
        if (data.optical_models.size() <= model_idx)
            data.optical_models.resize(model_idx + 1);
        return data.optical_models[model_idx];
    };

    auto set_mfp = [&](std::size_t mat_idx,
                       std::size_t model_idx,
                       std::pair<std::vector<double>, std::vector<double>> xy) {
        auto& mfp_table = model(model_idx).mfp_table;
        if (mfp_table.size() <= mat_idx)
            mfp_table.resize(mat_idx + 1);
        mfp_table[mat_idx]
            = native_physics_vector_from<units::Mev, units::Centimeter>(
                std::move(xy.first), std::move(xy.second));
    };

    model(0).model_class = ImportModelClass::absorption;
    model(1).model_class = ImportModelClass::rayleigh;
    model(2).model_class = ImportModelClass::wls;
    model(3).model_class = ImportModelClass::wls2;
    model(4).model_class = ImportModelClass::mie;

    // Material 0 + absorption model (index 0)
    mat(0).properties.refractive_index
        = native_physics_vector_from<units::ElectronVolt, units::Native>(
            {1.098177, 1.256172, 1.484130},
            {1.3235601610672, 1.3256740639273, 1.3280120256415});
    mat(0).rayleigh.scale_factor = 1;
    mat(0).rayleigh.compressibility
        = native_value_from(Compressibility{7.658e-23});
    mat(0).wls.mean_num_photons = 2;
    mat(0).wls.time_constant = native_value_from(TimeSecond(1e-9));
    mat(0).wls.component.x = {1.65e-6, 2e-6, 2.4e-6, 2.8e-6, 3.26e-6};
    mat(0).wls.component.y = {0.15, 0.25, 0.50, 0.40, 0.02};
    mat(0).wls2.mean_num_photons = 1;
    mat(0).wls2.time_constant = native_value_from(TimeSecond(21.7e-9));
    mat(0).wls2.component.x = {
        1.771e-6, 1.850e-6, 1.901e-6, 2.003e-6, 2.073e-6, 2.141e-6, 2.171e-6};
    mat(0).wls2.component.y = {0.016, 0.024, 0.040, 0.111, 0.206, 0.325, 0.413};
    mat(0).mie.forward_g = 0.99;
    mat(0).mie.backward_g = 0.99;
    mat(0).mie.forward_ratio = 0.8;
    set_mfp(0, 0, {{1e-3, 1e-2}, {5.7, 9.3}});
    set_mfp(0, 1, {{1e-2, 3e2}, {5.7, 9.3}});
    set_mfp(0, 2, {{1e-3, 2e-3, 5e-1}, {1.3, 4.9, 9.4}});
    set_mfp(0, 3, {{1e-1, 1e1}, {2.3, 5.4}});
    set_mfp(0, 4, {{1e-1, 1e1}, {2.3, 5.4}});

    // Material 1 + rayleigh model (index 1)
    mat(1).properties.refractive_index
        = native_physics_vector_from<units::ElectronVolt, units::Native>(
            {1.098177, 1.256172, 1.484130},
            {1.3235601610672, 1.3256740639273, 1.3280120256415});
    mat(1).rayleigh.scale_factor = 1.7;
    mat(1).rayleigh.compressibility
        = native_value_from(Compressibility{4.213e-24});
    set_mfp(1, 0, {{1e-2, 3e2}, {1.2, 10.7}});
    set_mfp(1, 1, {{1e-3, 1e-2}, {1.2, 10.7}});
    set_mfp(1, 2, {{1e-2, 3e2}, {5.7, 9.3}});
    set_mfp(1, 3, {{2e-2, 1e0, 3e2}, {5.7, 6.2, 9.3}});
    set_mfp(1, 4, {{2e-2, 1e0, 3e2}, {5.7, 6.2, 9.3}});

    // Material 2 + wls model (index 2)
    mat(2).properties.refractive_index
        = native_physics_vector_from<units::ElectronVolt, units::Native>(
            {1.098177, 6.812319}, {1.3235601610672, 1.4679465862259});
    mat(2).rayleigh.scale_factor = 1;
    mat(2).rayleigh.compressibility
        = native_value_from(Compressibility{7.658e-23});
    set_mfp(2, 0, {{1e-2, 3e2}, {3.1, 5.4}});
    set_mfp(2, 1, {{1e-3, 2e-3, 5e-1}, {0.1, 7.6, 12.5}});
    set_mfp(2, 2, {{1e-2, 3e2}, {1.2, 10.7}});
    set_mfp(2, 3, {{3e-2, 3e2}, {3.2, 9.4}});
    set_mfp(2, 4, {{3e-2, 3e2}, {3.2, 9.4}});

    // Material 3 + wls2 model (index 3)
    mat(3).properties.refractive_index
        = native_physics_vector_from<units::ElectronVolt, units::Native>(
            {1, 2, 5}, {1.3, 1.4, 1.5});
    mat(3).rayleigh.scale_factor = 2;
    mat(3).rayleigh.compressibility = native_value_from(Compressibility{1e-20});
    set_mfp(3, 0, {{2e-3, 5e1, 1e2}, {0.1, 7.6, 12.5}});
    set_mfp(3, 1, {{2e-3, 5e1, 1e2}, {0.1, 7.6, 12.5}});
    set_mfp(3, 2, {{2e-3, 5e1, 1e2}, {1.3, 4.9, 9.4}});
    set_mfp(3, 3, {{2e-3, 2e2}, {4.9, 9.4}});
    set_mfp(3, 4, {{2e-3, 2e2}, {4.9, 9.4}});

    // Material 4 + mie model (index 4)
    mat(4).properties.refractive_index
        = native_physics_vector_from<units::ElectronVolt, units::Native>(
            {1.098177, 6.812319}, {1.3235601610672, 1.4679465862259});
    mat(4).rayleigh.scale_factor = 1.7;
    mat(4).rayleigh.compressibility
        = native_value_from(Compressibility{4.213e-24});
    set_mfp(4, 0, {{1e-3, 2e-3, 5e-1}, {1.3, 4.9, 9.4}});
    set_mfp(4, 1, {{1e-3, 1e-2}, {3.1, 5.4}});
    set_mfp(4, 2, {{1e-3, 2e-3, 5e-1}, {1.3, 4.9, 9.4}});
    set_mfp(4, 3, {{1e-3, 4e-3, 5e-1}, {1.3, 5.9, 8.4}});
    set_mfp(4, 4, {{1e-3, 4e-3, 5e-1}, {1.3, 5.9, 8.4}});
}

//---------------------------------------------------------------------------//
/*!
 * Get the imported optical model corresponding to the given \c
 * ImportModelClass.
 */
auto OpticalMockTestBase::get_mfp_table(ImportModelClass imc) const
    -> VecGrid const&
{
    auto const& models = this->imported_data().optical_models;
    auto const iter = std::find_if(
        models.begin(), models.end(), [imc](ImportOpticalModel const& m) {
            return m.model_class == imc;
        });
    CELER_VALIDATE(iter != models.end(), << "invalid import model");

    return iter->mfp_table;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
