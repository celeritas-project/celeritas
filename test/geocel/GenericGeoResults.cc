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

void check_ref_eq(AssertionHelper& helper,
                  GenericGeoTrackingResult const& val1,
                  GenericGeoTrackingResult const& val2,
                  GenericGeoTrackingTolerance const& tol)
{
    is_vec_eq(helper, "volumes", val1.volumes, val2.volumes);
    is_vec_eq(helper,
              "volume_instances",
              val1.volume_instances,
              val2.volume_instances);
    is_vec_soft_eq(
        helper, "distances", val1.distances, val2.distances, tol.distance);
    if (val1.disabled_surface_normal() || val2.disabled_surface_normal())
    {
        static int warn_count{0};
        world_logger()(CELER_CODE_PROVENANCE,
                       warn_count++ == 0 ? LogLevel::warning : LogLevel::debug)
            << "Skipping surface normal comparison";
    }
    else
    {
        is_vec_soft_eq(
            helper, "dot_normal", val1.dot_normal, val2.dot_normal, tol.normal);
    }
    is_vec_soft_eq(helper,
                   "halfway_safeties",
                   val1.halfway_safeties,
                   val2.halfway_safeties,
                   EqualOr{SoftEqual(tol.safety, tol.safety)});
    is_vec_soft_eq(helper,
                   "bumps",
                   val1.bumps,
                   val2.bumps,
                   SoftEqual(tol.safety, tol.safety));
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

void check_ref_eq(AssertionHelper& helper,
                  GenericGeoVolumeStackResult const& val1,
                  GenericGeoVolumeStackResult const& val2)
{
    is_eq(helper,
          "volume_instances",
          val1.volume_instances,
          val2.volume_instances);
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

void check_ref_eq(AssertionHelper& helper,
                  GenericGeoModelInp const& val1,
                  GenericGeoModelInp const& val2)
{
#define IRE_COMPARE(ATTR) \
    if (val1.ATTR != val2.ATTR) \
    { \
        helper.fail() << "Expected " #ATTR ": " << repr(val1.ATTR) \
                      << " but got " << repr(val2.ATTR); \
    } \
    else \
        CELER_DISCARD(int)

    IRE_COMPARE(surface.labels);
    IRE_COMPARE(surface.volumes);
    IRE_COMPARE(region.labels);
    IRE_COMPARE(region.volumes);
    IRE_COMPARE(detector.labels);
    IRE_COMPARE(detector.volumes);

#undef IRE_COMPARE
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
