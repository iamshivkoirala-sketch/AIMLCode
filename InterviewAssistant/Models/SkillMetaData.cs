namespace InterviewAssistant.Models
{
    public class EmployeeMetaData
    {
        public string Name { get; set; }

        public string EmailAddress { get; set; }

        public string ContactNumber { get; set; }

        public string Location { get; set; }

        public string JobDescription { get; set; }

        public List<string> SkillSet { get; set; } = new();

        public string Education { get; set; }
    }
}
