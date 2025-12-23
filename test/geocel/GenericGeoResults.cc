//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoResults.cc
//---------------------------------------------------------------------------//
#include "GenericGeoResults.hh"

#include "corecel/OpaqueIdUtils.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/inp/Model.hh"

#include "GenericGeoTestInterface.hh"
#include "testdetail/TestMacrosImpl.hh"

//!@{
//! Helper macros
#define CELER_REF_ATTR(ATTR) "ref." #ATTR " = " << repr(this->ATTR) << ";\n"
//!@}

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
::testing::AssertionResult IsNormalEquiv(char const* expected_expr,
                                         char const* actual_expr,
                                         Real3 const& expected,
                                         Real3 const& actual)
{
    // Test that the normals are either in the same or opposite directions
    constexpr auto eps = SoftEqual<>{}.rel();
    if (norm(expected - actual) < eps || norm(expected + actual) < eps)
    {
        return ::testing::AssertionSuccess();
    }

    // Failed: print nice error message
    ::testing::AssertionResult result = ::testing::AssertionFailure();

    result << "Value of: " << actual_expr << "\n  Actual: " << repr(actual)
           << "\nExpected: " << expected_expr
           << "\nWhich is: " << repr(expected) << '\n';

    return result;
}

//---------------------------------------------------------------------------//
// TRACKING RESULT
//---------------------------------------------------------------------------//
// Replace dot-normals with a sentinel value
void GenericGeoTrackingResult::disable_surface_normal()
{
    this->dot_normal = {-2};
}

// Whether surface normals are disabled
bool GenericGeoTrackingResult::disabled_surface_normal() const
{
    auto& dn = this->dot_normal;
    return dn.size() == 1 && dn.front() == -2;
}

void GenericGeoTrackingResult::clear_boring_normals()
{
    auto& dn = this->dot_normal;
    if (std::all_of(dn.begin(), dn.end(), [](real_type n) {
            return soft_equal(n, real_type{1});
        }))
    {
        dn.clear();
    }
}

void GenericGeoTrackingResult::print_expected() const
{
    using std::cout;
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "GenericGeoTrackingResult ref;\n"
         << CELER_REF_ATTR(volumes) << CELER_REF_ATTR(volume_instances)
         << CELER_REF_ATTR(distances);
    if (this->dot_normal.empty())
    {
        // See clear_boring_normals
        cout << "ref.dot_normal = {}; // All normals are along track dir\n";
    }
    else if (this->disabled_surface_normal())
    {
        cout << "// Surface normal checking is disabled\n";
    }
    else
    {
        cout << CELER_REF_ATTR(dot_normal);
    }
    cout << CELER_REF_ATTR(halfway_safeties);
    if (!bumps.empty())
    {
        cout << CELER_REF_ATTR(bumps);
    }

    cout << "auto tol = test_->tracking_tol();\n"
            "EXPECT_REF_NEAR(ref, result, tol);\n"
            "/*** END CODE ***/\n";
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
    if (val1.disabled_surface_normal() || val2.disabled_surface_normal())
    {
        static int warn_count{0};
        world_logger()(CELER_CODE_PROVENANCE,
                       warn_count++ == 0 ? LogLevel::warning : LogLevel::debug)
            << "Skipping surface normal comparison";
    }
    else
    {
        IRE_VEC_SOFT_EQ(dot_normal, tol.normal);
    }
    IRE_VEC_SOFT_EQ(halfway_safeties,
                    EqualOr{SoftEqual(tol.safety, tol.safety)});
    IRE_VEC_SOFT_EQ(bumps, SoftEqual(tol.safety, tol.safety));

#undef IRE_VEC_EQ
#undef IRE_VEC_SOFT_EQ
    return helper;
}

//---------------------------------------------------------------------------//
// STACK RESULT
//---------------------------------------------------------------------------//
/*!
 * Construct a stack result from raw geometry output.
 */
GenericGeoVolumeStackResult
GenericGeoVolumeStackResult::from_span(LabelMap const& vol_inst,
                                       Span<VolumeInstanceId const> inst_ids)
{
    GenericGeoVolumeStackResult result;
    result.volume_instances.resize(inst_ids.size());
    for (auto i : range(inst_ids.size()))
    {
        auto vi_id = inst_ids[i];
        if (!vi_id)
        {
            result.volume_instances[i] = "<null>";
            continue;
        }
        result.volume_instances[i] = to_string(vol_inst.at(vi_id));
    }

    return result;
}

void GenericGeoVolumeStackResult::print_expected() const
{
    using std::cout;
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "GenericGeoVolumeStackResult ref;\n"
            << CELER_REF_ATTR(volume_instances)
            "EXPECT_REF_EQ(ref, result);\n"
            "/*** END CODE ***/\n";
}

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   GenericGeoVolumeStackResult const& val1,
                                   GenericGeoVolumeStackResult const& val2)
{
    AssertionHelper result{expr1, expr2};

#define IRE_COMPARE(ATTR)                                          \
    if (val1.ATTR != val2.ATTR)                                    \
    {                                                              \
        result.fail() << "Expected " #ATTR ": " << repr(val1.ATTR) \
                      << " but got " << repr(val2.ATTR);           \
    }
    IRE_COMPARE(volume_instances);
#undef IRE_COMPARE
    return result;
}

//---------------------------------------------------------------------------//
// MODEL INPUT RESULT
//---------------------------------------------------------------------------//
/*!
 * Construct a model input result from raw geometry model.
 */
GenericGeoModelInp GenericGeoModelInp::from_model_input(inp::Model const& in)
{
    GenericGeoModelInp result;

    // Extract volume data
    result.volume.labels.reserve(in.volumes.volumes.size());
    result.volume.materials.reserve(in.volumes.volumes.size());
    result.volume.daughters.reserve(in.volumes.volumes.size());

    for (auto i : range(in.volumes.volumes.size()))
    {
        auto const& vol = in.volumes.volumes[i];
        result.volume.labels.push_back(to_string(vol.label));
        result.volume.materials.push_back(id_to_int(vol.material));

        std::vector<int> daughters;
        daughters.reserve(vol.children.size());
        for (auto child_id : vol.children)
        {
            daughters.push_back(id_to_int(child_id));
        }
        result.volume.daughters.push_back(std::move(daughters));
    }

    // Extract volume instance data
    result.volume_instance.labels.reserve(in.volumes.volume_instances.size());
    result.volume_instance.volumes.reserve(in.volumes.volume_instances.size());

    for (auto i : range(in.volumes.volume_instances.size()))
    {
        auto const& vol_inst = in.volumes.volume_instances[i];
        result.volume_instance.labels.push_back(to_string(vol_inst.label));
        result.volume_instance.volumes.push_back(id_to_int(vol_inst.volume));
    }

    if (in.volumes.world < result.volume.labels.size())
    {
        result.world = result.volume.labels[in.volumes.world.get()];
    }
    else
    {
        result.world = "<invalid>";
    }

    // Extract surface data
    result.surface.labels.reserve(in.surfaces.surfaces.size());
    result.surface.volumes.reserve(in.surfaces.surfaces.size());

    for (auto const& surf : in.surfaces.surfaces)
    {
        result.surface.labels.push_back(to_string(surf.label));
        result.surface.volumes.push_back(std::visit(
            Overload{[](inp::Surface::Interface const& interface) {
                         return std::to_string(id_to_int(interface.first))
                                + "->"
                                + std::to_string(id_to_int(interface.second));
                     },
                     [](inp::Surface::Boundary const& boundary) {
                         return std::to_string(id_to_int(boundary));
                     }},
            surf.surface));
    }

    // Extract region data
    result.region.labels.reserve(in.regions.regions.size());
    result.region.volumes.reserve(in.regions.regions.size());
    for (auto const& reg : in.regions.regions)
    {
        result.region.labels.push_back(to_string(reg.label));
        std::vector<int> volumes;
        volumes.reserve(reg.volumes.size());
        for (auto vol_id : reg.volumes)
        {
            volumes.push_back(id_to_int(vol_id));
        }
        result.region.volumes.push_back(std::move(volumes));
    }

    // Extract detector data
    result.detector.labels.reserve(in.detectors.detectors.size());
    result.detector.volumes.reserve(in.detectors.detectors.size());
    for (auto const& det : in.detectors.detectors)
    {
        result.detector.labels.push_back(to_string(det.label));
        std::vector<int> volumes;
        volumes.reserve(det.volumes.size());
        for (auto vol_id : det.volumes)
        {
            volumes.push_back(id_to_int(vol_id));
        }
        result.detector.volumes.push_back(std::move(volumes));
    }

    return result;
}

void GenericGeoModelInp::print_expected() const
{
    using std::cout;
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "GenericGeoModelInp ref;\n"
         << CELER_REF_ATTR(volume.labels) << CELER_REF_ATTR(volume.materials)
         << CELER_REF_ATTR(volume.daughters)
         << CELER_REF_ATTR(volume_instance.labels)
         << CELER_REF_ATTR(volume_instance.volumes) << CELER_REF_ATTR(world);

    if (!surface.labels.empty())
    {
        cout << CELER_REF_ATTR(surface.labels)
             << CELER_REF_ATTR(surface.volumes);
    }
    if (!region.labels.empty())
    {
        cout << CELER_REF_ATTR(region.labels) << CELER_REF_ATTR(region.volumes);
    }
    if (!detector.labels.empty())
    {
        cout << CELER_REF_ATTR(detector.labels)
             << CELER_REF_ATTR(detector.volumes);
    }
    cout << "EXPECT_REF_EQ(ref, result);\n"
            "/*** END CODE ***/\n";
}

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   GenericGeoModelInp const& val1,
                                   GenericGeoModelInp const& val2)
{
    AssertionHelper result{expr1, expr2};

#define IRE_COMPARE(ATTR)                                          \
    if (val1.ATTR != val2.ATTR)                                    \
    {                                                              \
        result.fail() << "Expected " #ATTR ": " << repr(val1.ATTR) \
                      << " but got " << repr(val2.ATTR);           \
    }                                                              \
    else                                                           \
        (void)sizeof(char)

    IRE_COMPARE(volume.labels);
    IRE_COMPARE(volume.materials);
    IRE_COMPARE(volume.daughters);
    IRE_COMPARE(volume_instance.labels);
    IRE_COMPARE(volume_instance.volumes);
    IRE_COMPARE(world);
    IRE_COMPARE(surface.labels);
    IRE_COMPARE(surface.volumes);
    IRE_COMPARE(region.labels);
    IRE_COMPARE(region.volumes);
    IRE_COMPARE(detector.labels);
    IRE_COMPARE(detector.volumes);

#undef IRE_COMPARE
    return result;
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
