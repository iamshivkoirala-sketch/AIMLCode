using InterviewAssistant.LLMLayer;
using InterviewAssistant.Models;
using Microsoft.AspNetCore.Mvc;

namespace InterviewAssistant.Controllers
{
    public class ResumeController : Controller
    {
        public IActionResult Upload()
        {
            return View("UploadResume");
        }
        public async Task<IActionResult> SubmitResume(string resume)
        {
            var o = new Ollama();
            var t = await o.LoadResume(resume);
            
            return View("DisplaySkills" , t);
        }
        public async Task<IActionResult> SaveSkills(EmployeeMetaData obj)
        {
            
            return View("DisplaySkills", obj);
        }
    }
}
