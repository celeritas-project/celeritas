//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestInterface.cc
//---------------------------------------------------------------------------//
#include "GenericGeoTestInterface.hh"

#include <gtest/gtest.h>

#include "corecel/Config.hh"

#include "corecel/math/SoftEqual.hh"
#include "geocel/GeantGeoUtils.hh"

#if CELERITAS_USE_GEANT4
#    include <G4VPhysicalVolume.hh>

#    include "geocel/GeantGeoParams.hh"
#endif

namespace celeritas
{
namespace test
{

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

    auto const& volumes = this->geometry_interface()->impl_volumes();
    for (auto vidx : range(volumes.size()))
    {
        Label const& lab = volumes.at(ImplVolumeId{vidx});
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
    for (auto vidx : range(vol_inst.size()))
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
 *
 * TODO: clean this up and/or check that it's necessary when unifying volume
 * instances etc.
 */
std::vector<std::string> GenericGeoTestInterface::get_g4pv_labels() const
{
#if CELERITAS_USE_GEANT4
    auto geant_geo = celeritas::geant_geo().lock();
    CELER_VALIDATE(geant_geo, << "global Geant4 geometry is not loaded");

    auto& geo = *this->geometry_interface();
    auto const& vol_inst = geo.volume_instances();

    std::vector<std::string> result;
    for (auto vidx : range(vol_inst.size()))
    {
        VolumeInstanceId vi_id{vidx};
        if (vol_inst.at(vi_id).empty())
        {
            // Skip "virtual" PV
            continue;
        }

        result.push_back([&] {
            using namespace std::literals;

            auto phys_inst = geo.id_to_geant(vi_id);
            if (!phys_inst)
            {
                return "<null>"s;
            }

            auto vi_id = geant_geo->geant_to_id(*phys_inst.pv);
            auto const& vol_inst = geant_geo->volume_instances();
            if (!(vi_id < vol_inst.size()))
            {
                return "<out of range: "s + phys_inst.pv->GetName() + ">"s;
            }
            auto const& label = vol_inst.at(vi_id);
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
 * Get the volume name.
 */
std::string_view GenericGeoTestInterface::get_volume_name(ImplVolumeId i) const
{
    CELER_EXPECT(i);
    auto const& volumes = this->geometry_interface()->impl_volumes();
    if (i < volumes.size())
    {
        return volumes.at(ImplVolumeId{i.get()}).name;
    }
    return "<out of range>";
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
