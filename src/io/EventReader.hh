//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EventReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "physics/base/ParticleParams.hh"
#include "physics/base/Primary.hh"

namespace HepMC3
{
class Reader;
}

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read an event record file using the HepMC3 event record library and create
 * primary particles. Supported forrmats are Asciiv3, IO_GenEvent, HEPEVT, and
 * LHEF.
 */
class EventReader
{
  public:
    //!@{
    //! Type aliases
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    using result_type      = std::vector<Primary>;
    //!@}

  public:
    // Construct from a filename
    explicit EventReader(const char* filename, SPConstParticles params);

    // Default destructor in .cc
    ~EventReader();

    // Generate primary particles from the event record
    result_type operator()();

  private:
    // Shared standard model particle data
    SPConstParticles params_;

    // HepMC3 event record reader
    std::shared_ptr<HepMC3::Reader> input_file_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
