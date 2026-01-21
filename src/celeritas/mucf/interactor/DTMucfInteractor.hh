//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/interactor/DTMucfInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/StackAllocator.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/mucf/data/DTMixMucfData.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Muon-catalyzed fusion of \f$ (dt)_\mu \f$ molecules.
 *
 * Fusion channels:
 * - \f$ \alpha + \mu + n \f$
 * - \f$ (\alpha)_\mu + n \f$
 */
class DTMucfInteractor
{
  public:
    enum class Channel
    {
        alpha_muon_neutron,  //!< \f$ \alpha + \mu + n \f$
        muonicalpha_neutron,  //!< \f$ (\alpha)_\mu + n \f$
        size_
    };

    // Construct from shared and state data
    inline CELER_FUNCTION
    DTMucfInteractor(NativeCRef<DTMixMucfData> const& data,
                     Channel channel,
                     StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Shared constant physics properties
    NativeCRef<DTMixMucfData> const& data_;
    // Selected fusion channel
    Channel channel_{Channel::size_};
    // Allocate space for secondary particles
    StackAllocator<Secondary>& allocate_;
    // Number of secondaries per channel
    EnumArray<Channel, size_type> num_secondaries_{
        3,  // alpha_muon_neutron
        2  // muonicalpha_neutron
    };

    // Sample Interaction secondaries
    template<class Engine>
    inline CELER_FUNCTION Span<Secondary>
    sample_secondaries(Secondary* secondaries /*, other args */, Engine&);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
DTMucfInteractor::DTMucfInteractor(NativeCRef<DTMixMucfData> const& data,
                                   Channel const channel,
                                   StackAllocator<Secondary>& allocate)
    : data_(data), channel_(channel), allocate_(allocate)
{
    CELER_EXPECT(data_);
    CELER_EXPECT(channel_ < Channel::size_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a dt muonic molecule fusion.
 */
template<class Engine>
CELER_FUNCTION Interaction DTMucfInteractor::operator()(Engine& rng)
{
    // Allocate space for the final fusion channel
    Secondary* secondaries = allocate_(num_secondaries_[channel_]);
    if (secondaries == nullptr)
    {
        // Failed to allocate space for secondaries
        return Interaction::from_failure();
    }

    // Kill primary and generate secondaries
    Interaction result = Interaction::from_absorption();
    result.secondaries = this->sample_secondaries(secondaries, rng);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sample the secondaries of the selected channel.
 *
 * Since secondaries come from an at rest interaction, their final state is
 * a simple combination of random direction + momentum conservation
 */
template<class Engine>
CELER_FUNCTION Span<Secondary>
DTMucfInteractor::sample_secondaries(Secondary* secondaries /*, other args */,
                                     Engine&)
{
    // TODO: switch on channel_
    CELER_ASSERT_UNREACHABLE();
    return Span<Secondary>{secondaries, num_secondaries_[channel_]};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
