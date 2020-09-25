//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EventReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include "HepMC3/ReaderFactory.h"
#include "physics/base/ParticleParams.hh"
#include "physics/base/Primary.hh"

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
    //@{
    //! Type aliases
    using constSPParticleParams = std::shared_ptr<const ParticleParams>;
    using result_type           = std::vector<Primary>;
    //@}

  public:
    // Construct from a filename
    explicit EventReader(const char* filename, constSPParticleParams params);

    // Generate primary particles from the event record
    result_type operator()();

  private:
    // Shared standard model particle data
    constSPParticleParams params_;

    // HepMC3 event record reader
    std::shared_ptr<HepMC3::Reader> input_file_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
