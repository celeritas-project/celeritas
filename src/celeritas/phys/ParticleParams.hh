//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ParticleParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "PDGNumber.hh"
#include "ParticleData.hh"
#include "ParticleView.hh"

namespace celeritas
{
struct ImportData;
struct ImportParticle;

//---------------------------------------------------------------------------//
/*!
 * Data management for Standard Model particle classifications.
 *
 * This class represents "per-problem" shared data about standard
 * model particles being used.
 *
 * The ParticleParams is constructed on the host with a vector that
 * combines metadata (used for debugging output and interfacing with physics
 * setup) and data (used for on-device transport). Each entry in the
 * construction is assigned a unique \c ParticleId used for runtime access. See
 * \c celeritas::PDGNumber for details on the PDG code used during
 * construction.
 */
class ParticleParams final : public ParamsDataInterface<ParticleParamsData>
{
  public:
    //! Define a particle's input data
    struct ParticleInput
    {
        std::string name;  //!< Particle name
        PDGNumber pdg_code;  //!< See "Review of Particle Physics"
        units::MevMass mass;  //!< Rest mass [MeV / c^2]
        units::ElementaryCharge charge;  //!< Charge in units of [e]
        real_type decay_constant{};  //!< Decay constant [1/time]

        // Conversion function
        static ParticleInput from_import(ImportParticle const&);
    };

    //! Input data to construct this class
    using Input = std::vector<ParticleInput>;

  public:
    // Construct with imported data
    static std::shared_ptr<ParticleParams> from_import(ImportData const& data);

    // Construct with a vector of particle definitions
    explicit ParticleParams(Input const& defs);

    //// HOST ACCESSORS ////

    //! Number of particle definitions
    ParticleId::size_type size() const { return md_.size(); }

    // Get particle name
    inline std::string const& id_to_label(ParticleId id) const;

    // Get PDG code
    inline PDGNumber id_to_pdg(ParticleId id) const;

    // Find the ID from a name
    inline ParticleId find(std::string const& name) const;

    // Find the ID from a PDG code
    inline ParticleId find(PDGNumber pdg_code) const;

    // Access particle properties on host
    ParticleView get(ParticleId id) const;

    //! Access material properties on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access material properties on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Saved copy of metadata
    std::vector<std::pair<std::string, PDGNumber>> md_;

    // Map particle names to registered IDs
    std::unordered_map<std::string, ParticleId> name_to_id_;

    // Map particle codes to registered IDs
    std::unordered_map<PDGNumber, ParticleId> pdg_to_id_;

    // Host/device storage and reference
    CollectionMirror<ParticleParamsData> data_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get particle name.
 */
std::string const& ParticleParams::id_to_label(ParticleId id) const
{
    CELER_EXPECT(id < this->size());
    return md_[id.get()].first;
}

//---------------------------------------------------------------------------//
/*!
 * Get PDG code for a particle ID.
 */
PDGNumber ParticleParams::id_to_pdg(ParticleId id) const
{
    CELER_EXPECT(id < this->size());
    return md_[id.get()].second;
}

//---------------------------------------------------------------------------//
/*!
 * Find the ID from a name.
 */
ParticleId ParticleParams::find(std::string const& name) const
{
    auto iter = name_to_id_.find(name);
    if (iter == name_to_id_.end())
    {
        return ParticleId{};
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Find the ID from a PDG code.
 *
 * \todo Multiple particles can share a "generic" PDG code so this should be a
 * multimap.
 */
ParticleId ParticleParams::find(PDGNumber pdg_code) const
{
    auto iter = pdg_to_id_.find(pdg_code);
    if (iter == pdg_to_id_.end())
    {
        return ParticleId{};
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
