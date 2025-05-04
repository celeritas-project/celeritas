//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestInterface.cc
//---------------------------------------------------------------------------//
#include "GenericGeoTestInterface.hh"

#include <gtest/gtest.h>

#include "corecel/Config.hh"

#include "corecel/io/Repr.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/GeantGeoUtils.hh"

#include "testdetail/TestMacrosImpl.hh"

#if CELERITAS_USE_GEANT4
#    include <G4VPhysicalVolume.hh>
#endif

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
#define CELER_REF_ATTR(ATTR) "ref." #ATTR " = " << repr(this->ATTR) << ";\n"
void GenericGeoTrackingResult::print_expected()
{
    std::cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "GenericGeoTrackingResult ref;\n"
            CELER_REF_ATTR(volumes)
            CELER_REF_ATTR(volume_instances)
            CELER_REF_ATTR(distances)
            CELER_REF_ATTR(halfway_safeties)
            CELER_REF_ATTR(bumps)
            "auto tol = GenericGeoTrackingTolerance::from_test(*test_);\n"
            "EXPECT_RESULT_NEAR(ref, result, tol);\n"
            "/*** END CODE ***/\n";
}

GenericGeoTrackingTolerance
GenericGeoTrackingTolerance::from_test(GenericGeoTestInterface const& test)
{
    GenericGeoTrackingTolerance tol;
    tol.safety = test.safety_tol();
    tol.distance = SoftEqual{}.rel();
    return tol;
}

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   char const*,
                                   GenericGeoTrackingResult const& val1,
                                   GenericGeoTrackingResult const& val2,
                                   GenericGeoTrackingTolerance const& tol)
{
    using ::celeritas::testdetail::IsVecEq;
    using ::celeritas::testdetail::IsVecSoftEquiv;

    AssertionHelper helper{expr1, expr2};

#define IRE_VEC_EQ(ATTR)                                           \
    if (auto result = IsVecEq(expr1, #ATTR, val1.ATTR, val2.ATTR); \
        !static_cast<bool>(result))                                \
    {                                                              \
        helper.fail() << result.message();                         \
    }                                                              \
    else                                                           \
        (void)sizeof(char)
#define IRE_VEC_SOFT_EQ(ATTR, TOL)                                       \
    if (auto result                                                      \
        = IsVecSoftEquiv(expr1, #ATTR, #TOL, val1.ATTR, val2.ATTR, TOL); \
        !static_cast<bool>(result))                                      \
    {                                                                    \
        helper.fail() << result.message();                               \
    }                                                                    \
    else                                                                 \
        (void)sizeof(char)
#define IRE_VEC_SOFT_EQ2(ATTR, REL, ABS)                               \
    if (auto result = IsVecSoftEquiv(                                  \
            expr1, #ATTR, #REL, #ABS, val1.ATTR, val2.ATTR, REL, ABS); \
        !static_cast<bool>(result))                                    \
    {                                                                  \
        helper.fail() << result.message();                             \
    }                                                                  \
    else                                                               \
        (void)sizeof(char)

    IRE_VEC_EQ(volumes);
    IRE_VEC_EQ(volume_instances);
    IRE_VEC_SOFT_EQ(distances, tol.distance);
    IRE_VEC_SOFT_EQ2(halfway_safeties, tol.safety, tol.safety);
    IRE_VEC_SOFT_EQ2(bumps, tol.safety, tol.safety);

#undef IRE_COMPARE
    return helper;
}

//---------------------------------------------------------------------------//
void GenericGeoVolumeStackResult::print_expected()
{
    using std::cout;
    // clang-format off
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "GenericGeoVolumeStackResult ref;\n"
            CELER_REF_ATTR(volume_instances)
            CELER_REF_ATTR(replicas)
            "EXPECT_RESULT_EQ(ref, result);\n"
            "/*** END CODE ***/\n";
    // clang-format on
}

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   GenericGeoVolumeStackResult const& val1,
                                   GenericGeoVolumeStackResult const& val2)
{
    AssertionHelper result{expr1, expr2};

#define IRE_COMPARE(ATTR)                                           \
    if (val1.ATTR != val2.ATTR)                                     \
    {                                                               \
        result.fail() << "Actual " #ATTR ": " << repr(val1.ATTR) << " vs " \
               << repr(val2.ATTR);                                  \
    }                                                               \
    else                                                            \
        (void)sizeof(char)
    IRE_COMPARE(volume_instances);
    IRE_COMPARE(replicas);
#undef IRE_COMPARE
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the basename or unique geometry key (defaults to suite name).
 */
auto GenericGeoTestInterface::geometry_basename() const -> std::string
{
    // Get filename based on unit test name
    ::testing::TestInfo const* const test_info
        = ::testing::UnitTest::GetInstance()->current_test_info();
    CELER_ASSERT(test_info);
    return test_info->test_case_name();
}

//---------------------------------------------------------------------------//
/*!
 * Get the safety tolerance (defaults to SoftEq tol).
 */
real_type GenericGeoTestInterface::safety_tol() const
{
    constexpr SoftEqual<> default_seq{};
    return default_seq.rel();
}

//---------------------------------------------------------------------------//
/*!
 * Get the threshold for a movement being a "bump".
 *
 * This unitless tolerance is multiplied by the test's unit length when used.
 */
real_type GenericGeoTestInterface::bump_tol() const
{
    return 1e-7;
}

//---------------------------------------------------------------------------//
/*!
 * Get all logical volume labels.
 */
std::vector<std::string> GenericGeoTestInterface::get_volume_labels() const
{
    std::vector<std::string> result;

    auto const& volumes = this->geometry_interface()->volumes();
    for (auto vidx : range(this->volume_offset(), volumes.size()))
    {
        Label const& lab = volumes.at(VolumeId{vidx});
        if (!lab.empty())
        {
            result.emplace_back(to_string(lab));
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get all physical volume labels, including extensions.
 */
std::vector<std::string>
GenericGeoTestInterface::get_volume_instance_labels() const
{
    std::vector<std::string> result;

    auto const& vol_inst = this->geometry_interface()->volume_instances();
    for (auto vidx : range(this->volume_instance_offset(), vol_inst.size()))
    {
        Label const& lab = vol_inst.at(VolumeInstanceId{vidx});
        if (!lab.empty())
        {
            result.emplace_back(to_string(lab));
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get all Geant4 PV names corresponding to volume instances.
 */
std::vector<std::string> GenericGeoTestInterface::get_g4pv_labels() const
{
    auto* world = this->g4world();
    CELER_VALIDATE(world,
                   << "cannot get g4pv names from " << this->geometry_type()
                   << " geometry: Geant4 world has not been set");

#if CELERITAS_USE_GEANT4
    auto pv_labels = make_physical_vol_labels(*world);

    auto& geo = *this->geometry_interface();
    auto const& vol_inst = geo.volume_instances();

    std::vector<std::string> result;
    for (auto vidx : range(this->volume_instance_offset(), vol_inst.size()))
    {
        if (vol_inst.at(VolumeInstanceId{vidx}).empty())
        {
            // Skip "virtual" PV
            continue;
        }

        result.push_back([&] {
            using namespace std::literals;

            auto phys_inst = geo.id_to_geant(VolumeInstanceId{vidx});
            if (!phys_inst)
            {
                return "<null>"s;
            }
            auto id = static_cast<std::size_t>(phys_inst.pv->GetInstanceID());
            if (id >= pv_labels.size())
            {
                return "<out of range: "s + phys_inst.pv->GetName() + ">"s;
            }
            auto const& label = pv_labels[id];
            if (label.empty())
            {
                return "<not visited: "s + phys_inst.pv->GetName() + ">"s;
            }
            auto result = to_string(label);
            if (phys_inst.replica)
            {
                result += '@';
                result += std::to_string(phys_inst.replica.get());
            }
            return result;
        }());
    }
    return result;
#else
    CELER_NOT_CONFIGURED("Geant4");
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume name, adjusting for offsets from loading multiple geo.
 */
std::string_view GenericGeoTestInterface::get_volume_name(VolumeId i) const
{
    CELER_EXPECT(i);
    auto const& volumes = this->geometry_interface()->volumes();
    auto index = this->volume_offset() + i.get();
    if (index >= volumes.size())
    {
        return "<out of range>";
    }
    return volumes.at(VolumeId{index}).name;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
