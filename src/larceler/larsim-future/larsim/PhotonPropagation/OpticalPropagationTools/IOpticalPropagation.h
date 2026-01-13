//---------------------------------------------------------------------------//
//! \file IOpticalPropagation.h
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
//! \brief Abstract interface for optical simulation libraries
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>

namespace phot
{
class IOpticalPropagation;
}

//-------------------------------------------------------------------------//
/*!
 * Abstract interface for optical propagation.
 *
 * This interface allows the addition of different optical photon propagation
 * tools. As an \c art::tool, the \c executeEvent function is called *once*
 * during a \c art::EDProducer::produce module execution, and uses all \c
 * sim::SimEnergyDeposits from an \c art::Event found by the \c art::Handle .
 *
 * I.e. a single \c executeEvent function call propagates all resulting
 * optical photons from the existing batch of energy depositions on an
 * event-by-event basis. It is currently expected to manage 3 methods:
 * - \c PDFastSimPAR : already available in larsim
 * - \c Celeritas : Full optical particle transport on CPU and GPU
 * - \c Opticks : Full optical particle transport on Nvidia GPUs
 *
 * The interfaces takes a vector of \c sim::SimEnergyDeposit as input and
 * produces a vector of \c sim::OpDetBacktrackerRecord from detector hits.
 */
class phot::IOpticalPropagation
{
  public:
    //!@{
    //! \name Type aliases
    using VecSED = std::vector<sim::SimEnergyDeposit>;
    using UPVecBTR = std::unique_ptr<std::vector<sim::OpDetBacktrackerRecord>>;
    ///@}

    // Initialize tool
    virtual void beginJob() = 0;

    // Execute is called for every art::Event
    virtual UPVecBTR executeEvent(VecSED const& edeps) = 0;

    // Bring tool back to invalid state
    virtual void endJob() = 0;
};
