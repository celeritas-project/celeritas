//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/HitRootIO.cc
//---------------------------------------------------------------------------//
#include "HitRootIO.hh"

#include <regex>
#include <G4AutoDelete.hh>
#include <G4Event.hh>
#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <TBranch.h>
#include <TFile.h>
#include <TObject.h>
#include <TTree.h>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "accel/ExceptionConverter.hh"
#include "accel/SetupOptions.hh"

#include "GlobalSetup.hh"
#include "SensitiveHit.hh"

namespace
{
// Mutex within the HitRootIO scope
G4Mutex HitRootIOMutex = G4MUTEX_INITIALIZER;
}  // namespace

namespace demo_geant
{
G4ThreadLocal HitRootIO* HitRootIO::instance_ = nullptr;

//---------------------------------------------------------------------------//
/*!
 * Create a ROOT output file for each worker except the master thread if MT
 */
HitRootIO::HitRootIO()
{
    G4AutoLock lock(&HitRootIOMutex);
    file_name_ = std::regex_replace(
        GlobalSetup::Instance()->GetSetupOptions()->output_file,
        std::regex("json"),
        "root");

    if (G4Threading::IsWorkerThread())
    {
        file_name_ += std::to_string(G4Threading::G4GetThreadId());
    }

    if (G4Threading::IsWorkerThread()
        || !G4Threading::IsMultithreadedApplication())
    {
        file_ = TFile::Open(file_name_.c_str(), "recreate");
        CELER_VALIDATE(file_->IsOpen(), << "Failed to open " << file_name_);
        tree_ = new TTree(
            "Events", "Hit collections", this->SplitLevel(), file_);
        G4AutoDelete::Register(tree_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Return the static thread local singleton instance
 */
HitRootIO* HitRootIO::GetInstance()
{
    if (instance_ == nullptr)
    {
        static G4ThreadLocalSingleton<HitRootIO> instance;
        instance_ = instance.Instance();
    }
    return instance_;
}

//---------------------------------------------------------------------------//
/*!
 * Write sensitive hits to output in the form of HitRootEvent.
 */
void HitRootIO::WriteHits(G4Event const* event)
{
    G4HCofThisEvent* HCE = event->GetHCofThisEvent();
    if (HCE == nullptr)
    {
        return;
    }

    // Write the collection of sensitive hits into HitRootEvent
    auto hit_event = std::make_unique<HitRootEvent>();
    hit_event->SetEventID(event->GetEventID());
    HitRootEvent::HitContainer* hcmap = hit_event->GetHCMap();
    for (int i = 0; i < HCE->GetNumberOfCollections(); i++)
    {
        G4VHitsCollection* hc = HCE->GetHC(i);
        G4String hcname = hc->GetName();
        {
            std::vector<G4VHit*> hits;
            G4int number_of_hits = hc->GetSize();
            for (G4int j = 0; j < number_of_hits; ++j)
            {
                G4VHit* hit = hc->GetHit(j);
                SensitiveHit* sd_hit = dynamic_cast<SensitiveHit*>(hit);
                hits.push_back(sd_hit);
            }
            hcmap->insert(std::make_pair(hcname, hits));
        }
    }

    // Write a HitRootEvent into output ROOT file
    G4AutoLock lock(&HitRootIOMutex);
    this->WriteObject(hit_event.release());
}

//---------------------------------------------------------------------------//
/*!
 * Fill and write a HitRootEvent object
 */
void HitRootIO::WriteObject(HitRootEvent* hit_event)
{
    if (!init_branch_)
    {
        event_branch_ = tree_->Branch(
            "event.",
            &hit_event,
            GlobalSetup::Instance()->GetSetupOptions()->root_buffer_size,
            this->SplitLevel());
        init_branch_ = true;
    }

    tree_->Fill();
    file_->Write("", TObject::kOverwrite);
}

//---------------------------------------------------------------------------//
/*!
 * Close or Merge output
 */
void HitRootIO::Close()
{
    if (!G4Threading::IsMultithreadedApplication())
    {
        CELER_LOG(info) << "Writing hit ROOT output to " << file_name_ << "\"";
        file_ = tree_->GetCurrentFile();
        file_->Close();
    }
    else
    {
        // Merge output file on the master thread if MT
        if (G4Threading::IsMasterThread())
        {
            this->Merge();
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Merging output root files from multiple threads using TTree::MergeTrees
 *
 * TODO: use TBufferMerger and follow the example described in the ROOT
 * tutorials/multicore/mt103_fillNtupleFromMultipleThreads.C which stores
 * TBuffer data in memory and writes 32MB compressed output concurrently.
 */
void HitRootIO::Merge()
{
    auto nthreads = G4RunManager::GetRunManager()->GetNumberOfThreads();
    std::vector<TFile*> files;
    std::vector<TTree*> trees;
    std::unique_ptr<TList> list(new TList);

    celeritas::ExceptionConverter call_g4exception{"celer0006"};
    CELER_LOG(info) << "Merging hit root files from " << nthreads
                    << " threads into \"" << file_name_ << "\"";

    for (int i = 0; i < nthreads; ++i)
    {
        std::string file_name = file_name_ + std::to_string(i);
        files.push_back(TFile::Open(file_name.c_str()));
        trees.push_back((TTree*)(files[i]->Get("Events")));
        list->Add(trees[i]);

        if (i == nthreads - 1)
        {
            TFile* file = TFile::Open(file_name_.c_str(), "recreate");
            CELER_VALIDATE(file->IsOpen(), << "Failed to open " << file_name_);

            TTree* tree = TTree::MergeTrees(list.get());
            tree->SetName("Events");
            auto hit_event = std::make_unique<HitRootEvent>().release();
            tree->SetBranchAddress("event.", &hit_event);
            tree->Write();
            file->Close();
        }
        // Delete the merged file
        CELER_TRY_HANDLE(remove(file_name.c_str()), call_g4exception);
    }
}

//---------------------------------------------------------------------------//
}  // namespace demo_geant
