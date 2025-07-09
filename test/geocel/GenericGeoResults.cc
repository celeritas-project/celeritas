//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoResults.cc
//---------------------------------------------------------------------------//
#include "GenericGeoResults.hh"

#include "corecel/OpaqueIdUtils.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/io/Repr.hh"
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
// TRACKING RESULT
//---------------------------------------------------------------------------//

void GenericGeoTrackingResult::print_expected() const
{
    using std::cout;
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "GenericGeoTrackingResult ref;\n"
         << CELER_REF_ATTR(volumes) << CELER_REF_ATTR(volume_instances)
         << CELER_REF_ATTR(distances) << CELER_REF_ATTR(halfway_safeties)
         << CELER_REF_ATTR(bumps)
         << "auto tol = GenericGeoTrackingTolerance::from_test(*test_);\n"
            "EXPECT_REF_NEAR(ref, result, tol);\n"
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
                result.replicas[i] = id_to_int(phys_inst.replica);
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
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "GenericGeoVolumeStackResult ref;\n"
            << CELER_REF_ATTR(volume_instances)
            << CELER_REF_ATTR(replicas)
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
    IRE_COMPARE(replicas);
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
         << CELER_REF_ATTR(volume_instance.volumes);

    if (!surface.labels.empty())
    {
        cout << CELER_REF_ATTR(surface.labels)
             << CELER_REF_ATTR(surface.volumes);
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
    IRE_COMPARE(surface.labels);
    IRE_COMPARE(surface.volumes);

#undef IRE_COMPARE
    return result;
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
