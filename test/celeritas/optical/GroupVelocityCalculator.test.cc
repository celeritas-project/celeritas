//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/GroupVelocityCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/detail/GroupVelocityCalculator.hh"

#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/inp/Grid.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/optical/MaterialParams.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//

class GroupVelocityCalculatorTest : public ::celeritas::test::Test
{
  protected:
    std::shared_ptr<MaterialParams const> make_material(inp::Grid rindex)
    {
        ImportOpticalMaterial import_mat;
        import_mat.properties.refractive_index = std::move(rindex);

        MaterialParams::Input input;
        input.properties.push_back(std::move(import_mat.properties));
        input.volume_to_mat = {OptMatId{0}};
        input.optical_to_core = {PhysMatId{0}};

        return std::make_shared<MaterialParams const>(std::move(input));
    }
    inp::Grid make_refractive_index_grid()
    {
        inp::Grid result;
        result.x = {
            11.3512e-06, 11.3622e-06, 11.3732e-06, 11.3843e-06, 11.3954e-06,
            11.4065e-06, 11.4176e-06, 11.4288e-06, 11.44e-06,   11.4512e-06,
            11.4624e-06, 11.4736e-06, 11.4849e-06, 11.4962e-06, 11.5075e-06,
            11.5188e-06, 11.5302e-06, 11.5416e-06, 11.553e-06,  11.5644e-06,
            11.5758e-06,
        };
        // Refractive index
        result.y = {2.62477, 2.66804, 2.71432, 2.76397, 2.8174,  2.87508,
                    2.93758, 3.0056,  3.07994, 3.16162, 3.25186, 3.35224,
                    3.4647,  3.5918,  3.7369,  3.90453, 4.10102, 4.33545,
                    4.62145, 4.98065, 5.44987};

        return result;
    }
    inp::Grid make_refractive_index_water_grid()
    {
        inp::Grid result;
        result.x = {1e-07, 5e-07, 1.5e-06, 2.5e-06, 3.5e-06, 1e-5};
        // Refractive index
        result.y = {1, 1, 1.3333, 1.3333, 2, 2};

        return result;
    }
};

//---------------------------------------------------------------------------//
// Test group velocity for a refractive-index grid whose interval slopes
// increase rapidly at higher energies. This exercises the harmonic mean when
// one one-sided dn/dE is substantially larger than the other and verifies its
// effect on the calculated group velocity.
TEST_F(GroupVelocityCalculatorTest, host)
{
    auto rindex = this->make_refractive_index_grid();
    rindex.interpolation.type = InterpolationType::linear;
    auto material = this->make_material(std::move(rindex));

    detail::GroupVelocityCalculator calc{material->get(OptMatId{0})};

    std::vector<real_type> actual_group_velocity_over_c;

    // Photon energies
    static real_type const photon_energy[] = {
        11.4736e-06,
        11.4849e-06,
        11.4962e-06,
        11.5075e-06,
        11.5188e-06,
        11.5302e-06,
        11.5416e-06,
        11.5530e-06,
        11.5644e-06,
        11.5758e-06,
    };

    // Expected group velocities as dumped by Geant4
    // for the corresponding photon energies
    std::vector<real_type> expected_group_velocity_over_c = {
        0.008965113444663051,
        0.008015675516665738,
        0.007069333106999173,
        0.0061669453472211104,
        0.005332671683488536,
        0.004538329814472488,
        0.003770558509394347,
        0.0030546834027453218,
        0.00239375943059766,
        0.0020761055113155758};

    actual_group_velocity_over_c.reserve(std::size(photon_energy));

    for (auto i : range(std::size(photon_energy)))
    {
        real_type const group_vel = calc(units::MevEnergy{photon_energy[i]});
        actual_group_velocity_over_c.push_back(group_vel / constants::c_light);
    }

    EXPECT_VEC_NEAR(
        expected_group_velocity_over_c, actual_group_velocity_over_c, 1e-3);
}

//---------------------------------------------------------------------------//
// Test clamping at the lower endpoint of the refractive-index grid. Photon
// energies below the grid are clamped to its first energy, and group velocity
// is evaluated using the endpoint refractive index and the one-sided slope of
// its nearest interval.
TEST_F(GroupVelocityCalculatorTest, clamp)
{
    auto rindex = this->make_refractive_index_grid();
    rindex.interpolation.type = InterpolationType::linear;
    auto material = this->make_material(std::move(rindex));

    detail::GroupVelocityCalculator calc{material->get(OptMatId{0})};

    std::vector<real_type> actual_group_velocity_over_c;

    // photon energies
    static real_type const photon_energy[] = {
        10.7e-06,
        11.0e-6,
        11.3512e-06,
        11.3622e-06,
        11.3732e-06,
    };

    std::vector<real_type> expected_group_velocity_over_c = {
        0.021152264045848,
        0.021152264045848,
        0.021152264045848,
        0.0204645007297438,
        0.0192199927428051,
    };

    actual_group_velocity_over_c.reserve(std::size(photon_energy));

    for (auto i : range(std::size(photon_energy)))
    {
        real_type const group_vel = calc(units::MevEnergy{photon_energy[i]});
        actual_group_velocity_over_c.push_back(group_vel / constants::c_light);
    }

    EXPECT_VEC_SOFT_EQ(expected_group_velocity_over_c,
                       actual_group_velocity_over_c);
}

//---------------------------------------------------------------------------//
// Test a refractive-index grid with discontinuous slopes. The grid alternates
// between flat and rising linear intervals. Its harmonic-mean derivative is
// zero at every interior point because one adjacent slope is zero. Geant4
// instead constructs a GROUPVEL table using logarithmic finite differences at
// interval midpoints and then interpolates velocity, producing different
// results for the same refractive-index input.
TEST_F(GroupVelocityCalculatorTest, discontinuous_slope)
{
    auto rindex = this->make_refractive_index_water_grid();
    rindex.interpolation.type = InterpolationType::linear;
    auto material = this->make_material(std::move(rindex));

    detail::GroupVelocityCalculator calc{material->get(OptMatId{0})};

    std::vector<real_type> actual_group_velocity_over_c;

    // photon energies
    static real_type const photon_energy[] = {
        1e-06, 2e-6, 3e-06, 4e-06, 5e-06, 6e-06, 2e-07};

    // Reference values from Geant4's generated GROUPVEL table, retained here
    // to document the difference from the harmonic-derivative implementation.
    std::vector<real_type> expected_geant4_group_velocity_over_c = {
        0.6802569607544037,
        0.7500187504687618,
        0.2741159434659529,
        0.3063850943993882,
        0.33865424533282357,
        0.3709233962662588,
        1.0};

    // Expected Celeritas values
    std::vector<real_type> expected_group_velocity_over_c = {0.857155102215746,
                                                             0.7500187504687618,
                                                             0.600006000060001,
                                                             0.5,
                                                             0.5,
                                                             0.5,
                                                             1.0};
    actual_group_velocity_over_c.reserve(std::size(photon_energy));

    for (auto i : range(std::size(photon_energy)))
    {
        real_type const group_vel = calc(units::MevEnergy{photon_energy[i]});
        actual_group_velocity_over_c.push_back(group_vel / constants::c_light);
    }

    EXPECT_VEC_SOFT_EQ(expected_group_velocity_over_c,
                       actual_group_velocity_over_c);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
