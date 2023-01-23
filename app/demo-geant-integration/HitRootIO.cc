//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/HitRootIO.cc
//---------------------------------------------------------------------------//
#include "HitRootIO.hh"

#include <cstdio>
#include <regex>
#include <G4Event.hh>
#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <TBranch.h>
#include <TFile.h>
#include <TObject.h>
#include <TROOT.h>
#include <TTree.h>

#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "accel/ExceptionConverter.hh"
#include "accel/SetupOptions.hh"

#include "GlobalSetup.hh"
#include "SensitiveHit.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Create a ROOT output file for each worker except the master thread if MT
 */
HitRootIO::HitRootIO()
{
    ROOT::EnableThreadSafety();

    file_name_ = std::regex_replace(
        GlobalSetup::Instance()->GetSetupOptions()->output_file,
        std::regex("\\.json$"),
        ".root");

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
    }
}

//---------------------------------------------------------------------------//
/*!
 * Return the static thread local singleton instance
 */
HitRootIO* HitRootIO::GetInstance()
{
    static G4ThreadLocalSingleton<HitRootIO> instance;
    return instance.Instance();
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
        std::string hcname = hc->GetName();
        {
            std::vector<G4VHit*> hits;
            int number_of_hits = hc->GetSize();
            for (int j = 0; j < number_of_hits; ++j)
            {
                G4VHit* hit = hc->GetHit(j);
                SensitiveHit* sd_hit = dynamic_cast<SensitiveHit*>(hit);
                hits.push_back(sd_hit);
            }
            hcmap->insert(std::make_pair(hcname, hits));
        }
    }

    // Write a HitRootEvent into output ROOT file
    this->WriteObject(hit_event.release());
}

//---------------------------------------------------------------------------//
/*!
 * Fill a HitRootEvent object
 */
void HitRootIO::WriteObject(HitRootEvent* hit_event)
{
    if (!init_branch_)
    {
        event_branch_
            = tree_->Branch("event.",
                            &hit_event,
                            GlobalSetup::Instance()->GetRootBufferSize(),
                            this->SplitLevel());
        init_branch_ = true;
    }
    else
    {
        event_branch_->SetAddress(&hit_event);
    }

    tree_->Fill();
    event_branch_->ResetAddress();
}

//---------------------------------------------------------------------------//
/*!
 * Write, and Close or Merge output
 */
void HitRootIO::Close()
{
    if (!G4Threading::IsMultithreadedApplication())
    {
        CELER_LOG(info) << "Writing hit ROOT output to " << file_name_ << "\"";
        file_ = tree_->GetCurrentFile();
        file_->Write("", TObject::kOverwrite);
        file_->Close();
    }
    else
    {
        // Merge output file on the master thread if MT
        if (G4Threading::IsMasterThread())
        {
            this->Merge();
        }
        else
        {
            file_->Write("", TObject::kOverwrite);
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
            //  Write both the TFile and TTree meta-data
            file->Write();
            file->Close();
        }
        // Delete the merged file
        CELER_TRY_HANDLE(std::remove(file_name.c_str()), call_g4exception);
    }
}

//---------------------------------------------------------------------------//
}  // namespace demo_geant
