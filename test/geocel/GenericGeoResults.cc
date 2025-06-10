//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoResults.cc
//---------------------------------------------------------------------------//
#include "GenericGeoResults.hh"

#include "corecel/io/Repr.hh"
#include "corecel/math/SoftEqual.hh"

#include "GenericGeoTestInterface.hh"
#include "testdetail/TestMacrosImpl.hh"

// DEPRECATED: remove in v0.7
#define EXPECT_RESULT_EQ(EXPECTED, ACTUAL) EXPECT_REF_EQ(EXPECTED, ACTUAL)
#define EXPECT_RESULT_NEAR(EXPECTED, ACTUAL, TOL) \
    EXPECT_REF_NEAR(EXPECTED, ACTUAL, TOL)

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TRACKING RESULT
//---------------------------------------------------------------------------//

#define CELER_REF_ATTR(ATTR) "ref." #ATTR " = " << repr(this->ATTR) << ";\n"
void GenericGeoTrackingResult::print_expected() const
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

    IRE_VEC_EQ(volumes);
    IRE_VEC_EQ(volume_instances);
    IRE_VEC_SOFT_EQ(distances, tol.distance);
    IRE_VEC_SOFT_EQ(halfway_safeties, SoftEqual(tol.safety, tol.safety));
    IRE_VEC_SOFT_EQ(bumps, SoftEqual(tol.safety, tol.safety));

#undef IRE_COMPARE
    return helper;
}

//---------------------------------------------------------------------------//
// STACK RESULT
//---------------------------------------------------------------------------//
/*!
 * Construct a stack result from raw geometry output.
 */
GenericGeoVolumeStackResult
GenericGeoVolumeStackResult::from_span(GeoParamsInterface const& geo,
                                       Span<VolumeInstanceId const> inst_ids)
{
    auto const& vol_inst = geo.volume_instances();

    GenericGeoVolumeStackResult result;
    result.volume_instances.resize(inst_ids.size());
    result.replicas.assign(inst_ids.size(), -1);
    for (auto i : range(inst_ids.size()))
    {
        auto vi_id = inst_ids[i];
        if (!vi_id)
        {
            result.volume_instances[i] = "<null>";
            continue;
        }
        auto const& label = vol_inst.at(vi_id);
        if (auto phys_inst = geo.id_to_geant(vi_id))
        {
            if (phys_inst.replica)
            {
                result.replicas[i] = phys_inst.replica.get();
            }
        }
        // Only write extension if not a replica, because Geant4
        // effectively gives multiple volume instances the same name+ext
        result.volume_instances[i] = !result.replicas[i] ? to_string(label)
                                                         : label.name;
    }

    return result;
}

void GenericGeoVolumeStackResult::print_expected() const
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

#define IRE_COMPARE(ATTR)                                                  \
    if (val1.ATTR != val2.ATTR)                                            \
    {                                                                      \
        result.fail() << "Actual " #ATTR ": " << repr(val1.ATTR) << " vs " \
                      << repr(val2.ATTR);                                  \
    }                                                                      \
    else                                                                   \
        (void)sizeof(char)
    IRE_COMPARE(volume_instances);
    IRE_COMPARE(replicas);
#undef IRE_COMPARE
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
