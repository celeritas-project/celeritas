//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoResults.cc
//---------------------------------------------------------------------------//
#include "GenericGeoResults.hh"

#include <algorithm>

#include "corecel/OpaqueIdUtils.hh"
#include "corecel/Types.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/inp/Model.hh"

#include "GenericGeoTestInterface.hh"
#include "RefComparer.hh"
#include "TestMacros.hh"

using namespace celeritas::literals;

namespace celeritas
{
namespace test
{
namespace
{

template<class T>
void erase_after(std::vector<T>& vec, std::size_t idx)
{
    auto iter = vec.begin() + std::min(idx, vec.size());
    vec.erase(iter, vec.end());
}

}  // namespace

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
            return soft_equal(n, 1_r);
        }))
    {
        dn.clear();
    }
}

void GenericGeoTrackingResult::fail_at(std::size_t index)
{
    erase_after(volumes, index);
    erase_after(volume_instances, index);
    erase_after(distances, index);
    erase_after(dot_normal, index);
    erase_after(halfway_safeties, index);
    volumes.push_back("[FAILURE]");
}

std::ostream& operator<<(std::ostream& os, GenericGeoTrackingResult const& ref)
{
    return os << '{' << CELER_ATTR_REPR(ref, volumes) << ", "
              << CELER_ATTR_REPR(ref, volume_instances) << ", "
              << CELER_ATTR_REPR(ref, distances) << ", "
              << CELER_ATTR_REPR(ref, dot_normal) << ", "
              << CELER_ATTR_REPR(ref, halfway_safeties) << ", "
              << CELER_ATTR_REPR(ref, bumps) << '}';
}

bool is_ref_eq(AssertionHelper& helper,
               GenericGeoTrackingResult const& val1,
               GenericGeoTrackingResult const& val2,
               GenericGeoTrackingTolerance const& tol)
{
    RefComparer<GenericGeoTrackingResult> compare{helper, val1, val2};
    CELER_INVOKE_ATTR(compare.is_eq, GenericGeoTrackingResult, volumes);
    CELER_INVOKE_ATTR(
        compare.is_eq, GenericGeoTrackingResult, volume_instances);
    CELER_INVOKE_ATTR(
        compare.is_soft_eq, GenericGeoTrackingResult, distances, tol.distance);
    if (val1.disabled_surface_normal() || val2.disabled_surface_normal())
    {
        static int warn_count{0};
        world_logger()(CELER_CODE_PROVENANCE,
                       warn_count++ == 0 ? LogLevel::warning : LogLevel::debug)
            << "Skipping surface normal comparison";
    }
    else
    {
        CELER_INVOKE_ATTR(compare.is_soft_eq,
                          GenericGeoTrackingResult,
                          dot_normal,
                          tol.normal);
    }
    CELER_INVOKE_ATTR(compare.is_soft_eq,
                      GenericGeoTrackingResult,
                      halfway_safeties,
                      EqualOr{SoftEqual(tol.safety, tol.safety)});
    CELER_INVOKE_ATTR(compare.is_soft_eq,
                      GenericGeoTrackingResult,
                      bumps,
                      SoftEqual(tol.safety, tol.safety));
    return static_cast<bool>(compare);
}

//---------------------------------------------------------------------------//
// STACK RESULT
//---------------------------------------------------------------------------//
/*!
 * Construct a stack result from raw geometry output.
 */
GenericGeoVolumeStackResult GenericGeoVolumeStackResult::from_span(
    LabelMap const& vol_inst, Span<VolumeInstanceId const> inst_ids)
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

std::ostream& operator<<(std::ostream& os,
                         GenericGeoVolumeStackResult const& ref)
{
    return os << '{' << CELER_ATTR_REPR(ref, volume_instances) << '}';
}

void GenericGeoVolumeStackResult::fail()
{
    this->volume_instances.push_back("[FAILURE]");
}

bool is_ref_eq(AssertionHelper& helper,
               GenericGeoVolumeStackResult const& val1,
               GenericGeoVolumeStackResult const& val2)
{
    RefComparer<GenericGeoVolumeStackResult> compare{helper, val1, val2};
    CELER_INVOKE_ATTR(
        compare.is_eq, GenericGeoVolumeStackResult, volume_instances);
    return static_cast<bool>(compare);
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

std::ostream& operator<<(std::ostream& os, GenericGeoModelInp const& ref)
{
    return os << "{{" << CELER_ATTR_REPR(ref, surface.labels) << ", "
              << CELER_ATTR_REPR(ref, surface.volumes) << "}, {"
              << repr(ref.region.labels) << ", "
              << CELER_ATTR_REPR(ref, region.volumes) << "}, {"
              << CELER_ATTR_REPR(ref, detector.labels) << ", "
              << CELER_ATTR_REPR(ref, detector.volumes) << "}}";
}

bool is_ref_eq(AssertionHelper& helper,
               GenericGeoModelInp const& val1,
               GenericGeoModelInp const& val2)
{
    RefComparer<GenericGeoModelInp> compare{helper, val1, val2};
    auto compare_attr = [&helper, &compare](char const* name,
                                            auto const& expected,
                                            auto const& actual) {
        bool const result = expected == actual;
        if (!result)
        {
            helper.fail() << "Expected " << name << ": " << repr(expected)
                          << " but got " << repr(actual);
        }
        compare.record_result(result);
    };

    compare_attr("surface.labels", val1.surface.labels, val2.surface.labels);
    compare_attr("surface.volumes", val1.surface.volumes, val2.surface.volumes);
    compare_attr("region.labels", val1.region.labels, val2.region.labels);
    compare_attr("region.volumes", val1.region.volumes, val2.region.volumes);
    compare_attr("detector.labels", val1.detector.labels, val2.detector.labels);
    compare_attr(
        "detector.volumes", val1.detector.volumes, val2.detector.volumes);
    return static_cast<bool>(compare);
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
